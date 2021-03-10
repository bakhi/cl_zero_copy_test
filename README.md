A simple test for OpenCL zero-copy operation using vector addition.

## NOTE

clEnqueueWriteBuffer: Enqueue commands to write to a buffer object from host memory.

clEnqueueMapBuffer: Enqueues a command to map a region of the buffer object given by buffer into the host address space and returns a pointer to this mapped region.

The latter results in zero-copy by mapping device memory to the host address space. 
There is tradeoff between them: zero-copy but more CL commands vs. copy overhead but less CL commands

## Build
`./build.sh`
