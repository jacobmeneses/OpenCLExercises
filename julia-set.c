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
  FILE *fp, *fout;
  char fileName[] = "./julia-set.cl";
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
  long _size;
  long localSize;
  size_t global_work_size[3], local_work_size[3];
  cl_event event;
  float width_f, height_f;

  int i,d,l,x,y, width, height, N;

  width = 1024;
  height = 1024;
  width_f = (float) width;
  height_f = (float) height;
  N = width * height * 3;
  output_numbers = (float*) malloc(sizeof(float) * N);
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

  /* Load the source code containing the kernel */
  fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }

  source_str = (char*) malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  /* Create OpenCL context */
  context = clCreateContext(NULL, 1, &selected_device_id, NULL, NULL, &ret);
  /* Create Command Queue */
  command_queue = clCreateCommandQueue(context, selected_device_id, 0, &ret);

  /* Create Memory Buffer */
  output_numbers_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &ret);

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
  kernel = clCreateKernel(program, "julia", &ret);
  if(ret != CL_SUCCESS) { printf("Error creating kernel\n"); }

  /* Set OpenCL Kernel Parameters */
  ret = clSetKernelArg(kernel, 0, sizeof(float), (void *)&width_f);
  ret = clSetKernelArg(kernel, 1, sizeof(float), (void *)&height_f);
  ret = clSetKernelArg(kernel, 2, sizeof(int), (void *)&width);
  ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&height);
  ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&output_numbers_obj);

  global_work_size[0] = width;
  global_work_size[1] = height;
  local_work_size[0] = 16;
  local_work_size[1] = 16;

  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
  if(ret != CL_SUCCESS) { printf("ERROR executing kernel kernel %d\n", ret); }

  clWaitForEvents(1, &event);

  /* Read final results */
  ret = clEnqueueReadBuffer(command_queue, output_numbers_obj, CL_TRUE, 0, sizeof(float)* N, output_numbers, 0, NULL, NULL);
  if(ret != CL_SUCCESS) { printf("ERROR reading buffer %d\n", ret); }

  fout = fopen("file.ppm", "w");
  fprintf(fout, "P3\n%d %d\n", width, height);
  for(x = 0 ; x < width; x++) {
    for(y = 0 ; y < height; y++) {
        fprintf(fout, "%d %d %d ", 
        (int) output_numbers[x*width+y],
        (int) output_numbers[x*width+y+1],
        (int) output_numbers[x*width+y+2]
        );

    }
    fprintf(fout, "\n");
  }
  fclose(fout);

  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(output_numbers_obj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(output_numbers);
}
