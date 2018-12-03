__kernel void sum(
  __global float* numbers,
  __global float* output,
  __local float* localReductions)
{
  const int globalId = get_global_id(0);
  const int localId = get_local_id(0);
  const int localSize = get_local_size(0);
  const int workgroupId = globalId / localSize;

  localReductions[localId] = numbers[globalId];

  for(int offset = localSize / 2; offset > 0; offset /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localId < offset) {
      localReductions[localId] += localReductions[localId + offset];
    }
  }

  if(localId == 0) {
    output[workgroupId] = localReductions[0];
  }
}
