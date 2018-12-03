#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

void print_device_info(cl_device_id device_id){
  char name[255];
  cl_uint max_compute_units;
  cl_ulong global_memory_size;
  cl_ulong local_memory_size;
  size_t max_work_group_size;
  size_t max_work_item_sizes[3];
  cl_uint max_work_item_dimensions;
  cl_int ret;

  clGetDeviceInfo(device_id, CL_DEVICE_NAME, 255, &name, NULL);
  printf("Device name: %s\n", name);
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, NULL);
  printf("\tMax compute units: %d\n", max_compute_units);
  clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_memory_size, NULL);
  printf("\tGlobal memory size: %llu (%llu mb)\n", global_memory_size, (global_memory_size/ 1000000));
  clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_memory_size, NULL);
  printf("\tLocal memory size: %llu (%llu kb)\n", local_memory_size, (local_memory_size/1000));

  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
  printf("\tMax work group size: %lu\n", max_work_group_size);

  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dimensions, NULL);
  printf("\tMax work item dimensions: %d\n", max_work_item_dimensions);

  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, &max_work_item_sizes, NULL);
  printf("\tMax work item sizes: (%lu, %lu, %lu)\n", max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);

  printf("How many floats per local_memory_size = %llu\n", local_memory_size / sizeof(float));
}

int main() {
  cl_device_id device_id = NULL;
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;
  cl_platform_id platforms[20];
  cl_device_id devices[20];
  cl_device_type device_type;
  cl_device_id selected_device_id = NULL;

  char string[MEM_SIZE];
  FILE *fp;
  char fileName[] = "./work-groups.cl";
  char *source_str;
  size_t source_size;
  cl_mem numbers_obj = NULL;
  cl_mem output_numbers_obj = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  float *numbers;
  float *grouped;
  float *output_numbers;
  float rand_max = (float)RAND_MAX;
  float rnumber;
  long _size;
  long localSize;
  size_t global_work_size[3], local_work_size[3];
  cl_event event;
  float total_sum_host;
  float total_sum_host_kahan;
  float total_sum_device;
  float total_sum_host_grouped;
  float total_sum_host_grouped_kahan;
  float _c, _y, _t, _sum, _x;

  int i,d,l,x;
  int N = 1048576; // 2 power 20;

  /* Get Platform and Device Info */
  ret = clGetPlatformIDs(20, platforms, &ret_num_platforms);

  for(i = 0; i < ret_num_platforms ; i++) {
    platform_id = platforms[i];
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 20, devices, &ret_num_devices);

    for(d = 0 ; d < ret_num_devices; d++) {
      device_id = devices[d];

      clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);

      if( device_type == CL_DEVICE_TYPE_GPU ) {
        print_device_info(device_id);
        selected_device_id = device_id;
      }
    }
  }

  printf("Number of available platforms = %d\n", ret_num_platforms);
  _size = sizeof(float) * N;
  printf("Memory in bytes of numbers = %lu (%lu kb)\n", _size, (_size/1000) );

  /* Load the source code containing the kernel */
  fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }

  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  printf("Generating host data...\n");
  numbers = (float*) malloc(sizeof(float) * N);
  grouped = (float*) malloc(sizeof(float) * N);
  output_numbers = (float*) malloc(sizeof(float) * N);
  srand(10);

  total_sum_host = 0.0f;
  for(i = 0 ; i < N; i++){
    rnumber = (float)rand() / rand_max;
    // rnumber = 1.0f;
    numbers[i] = rnumber;
    total_sum_host += rnumber;
  }

  /* computing sum kahan algorithm */
  total_sum_host_kahan = 0.0f;
  _c = 0.0f;
  for(i = 0 ; i < N; i++){
    _y = numbers[i] - _c;
    _t = total_sum_host_kahan + _y;
    _c = (_t - total_sum_host_kahan) - _y;
    total_sum_host_kahan = _t;
  }
  
  d = (1024 * 1024) / 256; 

  /* Computing sum like device would do */
  for(i = 0; i < d; i++){
    grouped[i] = 0.0f;
    for(x = 0; x < 256; x++){
      grouped[i] += numbers[i * 256 + x];
    }
  }
  total_sum_host_grouped = 0.0f;
  for(i = 0; i < d; i++){
    total_sum_host_grouped += grouped[i];
  }

  /* Computing sum like device would do (kahan algorithm)*/
  for(i = 0; i < d; i++){
    grouped[i] = 0.0f;
    _c = 0.0f;
    for(x = 0; x < 256; x++){
      // kahan
      _y = numbers[i * 256 + x] - _c;
      _t = grouped[i] + _y;
      _c = (_t - grouped[i]) - _y;
      grouped[i]= _t;
    }
  }
  total_sum_host_grouped_kahan = 0.0f;
  _c = 0.0f;
  for(i = 0; i < d; i++){
    // kahan
    _y = grouped[i] - _c;
    _t = total_sum_host_grouped_kahan + _y;
    _c = (_t - total_sum_host_grouped_kahan) - _y;
    total_sum_host_grouped_kahan = _t;
  }

  printf("Total of numbers generated = %d\n", N);
  printf("Sum of all numbers (computed on host) = %f\n", total_sum_host);
  printf("Sum of all numbers (computed on host, kahan algorithm) = %f\n", total_sum_host_kahan);
  printf("Sum of all numbers grouped (the device way on host) = %f\n", total_sum_host_grouped);
  printf("Sum of all numbers grouped (the device way on host, kahan algorithm) = %f\n", total_sum_host_grouped_kahan);

  /* Create OpenCL context */
  context = clCreateContext(NULL, 1, &selected_device_id, NULL, NULL, &ret);
  /* Create Command Queue */
  command_queue = clCreateCommandQueue(context, selected_device_id, 0, &ret);

  /* Create Memory Buffer */
  numbers_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &ret);
  output_numbers_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &ret);
  /* Writting the input buffer */
  ret = clEnqueueWriteBuffer(command_queue, numbers_obj, CL_TRUE, 0, sizeof(float)*N, numbers, 0, NULL, NULL);

  /* Create Kernel Program from the source */
  program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
  /* Build Kernel Program */
  ret = clBuildProgram(program, 1, &selected_device_id, NULL, NULL, NULL);

  if(ret != CL_SUCCESS) {
    puts("Build program error");
    size_t len;
    char buildLog[2048];
    clGetProgramBuildInfo(program, selected_device_id, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, &len);
    printf("%s\n", buildLog);
  }

  /* Create OpenCL Kernel */
  kernel = clCreateKernel(program, "sum", &ret);
  if(ret != CL_SUCCESS) { printf("Error creating kernel\n"); }

  /* Set OpenCL Kernel Parameters */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&numbers_obj);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_numbers_obj);
  ret = clSetKernelArg(kernel, 2, sizeof(float) * (1024 * 1024) / 256, NULL);

  global_work_size[0] = 1024 * 1024;
  local_work_size[0] = 256;
  printf("Size expected of local memory = %lu\n", 256 * sizeof(float));

  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event);
  if(ret != CL_SUCCESS) { printf("ERROR executing kernel kernel %d\n", ret); }

  clWaitForEvents(1, &event);

  /* Read final results */
  ret = clEnqueueReadBuffer(command_queue, output_numbers_obj, CL_TRUE, 0, sizeof(float)*N, numbers, 0, NULL, NULL);
  if(ret != CL_SUCCESS) { printf("ERROR reading buffer %d\n", ret); }

  total_sum_device = 0.0f;
  d = (1024 * 1024) / 256;
  for(i = 0; i < d; i++){
    total_sum_device += numbers[i];
  }
  printf("Sum of all numbers (computed on device, reduced on host) = %f\n", total_sum_device);

  // l = 4096;
  // l = 1;
  // printf("%f %f %f %f\n", numbers[l - 1], numbers[l], numbers[l + 1], numbers[255]);

  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(numbers_obj);
  ret = clReleaseMemObject(output_numbers_obj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(numbers);
  free(grouped);
  free(output_numbers);
}
