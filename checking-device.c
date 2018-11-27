#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

int main() {
  cl_device_id device_id = NULL;
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;
  cl_platform_id platforms[20];
  cl_device_id devices[20];
  cl_uint max_compute_units;
  cl_ulong global_memory_size;
  cl_ulong local_memory_size;
  size_t max_work_group_size;
  cl_uint max_work_item_dimensions;
  size_t max_work_item_sizes[3];

  int i,d;
  char name[255];

  /* Get Platform and Device Info */
  ret = clGetPlatformIDs(20, platforms, &ret_num_platforms);

  // ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

  for(i = 0; i < ret_num_platforms ; i++) {
    platform_id = platforms[i];
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 20, devices, &ret_num_devices);

    for(d = 0 ; d < ret_num_devices; d++) {
      device_id = devices[d];

      clGetDeviceInfo(device_id, CL_DEVICE_NAME, 255, &name, NULL);
      printf("Device name: %s\n", name);
      clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, NULL);
      printf("\tMax compute units: %d\n", max_compute_units);
      clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_memory_size, NULL);
      printf("\tGlobal memory size: %llu\n", global_memory_size);
      clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_memory_size, NULL);
      printf("\tLocal memory size: %llu\n", local_memory_size);

      clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
      printf("\tMax work group size: %lu\n", max_work_group_size);

      clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
      printf("\tMax work group size: %lu\n", max_work_group_size);

      clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dimensions, NULL);
      printf("\tMax work item dimensions: %d\n", max_work_item_dimensions);

      clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, &max_work_item_sizes, NULL);
      printf("\tMax work item sizes: (%lu, %lu, %lu)\n", max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);
    }
  }

  printf("Number of available platforms = %d\n", ret_num_platforms);
}
