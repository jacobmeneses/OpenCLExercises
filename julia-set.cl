#define cX -0.7
#define cY 0.27015

__kernel void julia(
  float width_f,
  float height_f,
  int width,
  int height,
  __global float* output
  )
{
    // y, x
    int globalIdX = get_global_id(0);
    int globalIdY = get_global_id(1);
    int i;

    float x = (float) globalIdX;
    float y = (float) globalIdY;
    float tmp;

    float zx = 1.5 * (x - width_f/2) / (0.5 * width_f);
    float zy = (y - height_f/2) / (0.5 * height_f);
    i = 255;
    while (zx*zx+zy*zy < 4.0 && i > 0) {
        tmp = zx*zx - zy*zy + cX;
        zy = 2.0*zx*zy + cY;
        zx = tmp;
        i--;
    }

    output[globalIdX * width + globalIdY ] =  (float)i;
    output[globalIdX * width + globalIdY + 1] = (float) i;
    output[globalIdX * width + globalIdY + 2] =  (float) (i << 3);
}

