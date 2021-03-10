#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/*
 * FYI - 	https://downloads.ti.com/mctools/esd/docs/opencl/memory/host-malloc-extension.html
 * 			https://downloads.ti.com/mctools/esd/docs/opencl/memory/access-model.html
 */

#define MAX_SOURCE_SIZE (0x100000)
#define NRUNS 1

// toggle this to select ZERO_COPY 
#define ZERO_COPY

int run() {
	int i;
//	const int LIST_SIZE = 64;
	const int LIST_SIZE = 32 * 1024 * 1024;
	FILE *fp;
	char *source_str;
	size_t source_size;

	// Load the kernel source code into the array source_str
	fp = fopen("vector_add_kernel.cl", "r");
	if (!fp) {
			fprintf(stderr, "Failed to load kernel.\n");
			exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;   
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, 
					&device_id, &ret_num_devices);

	// Create an OpenCL context
	cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, 
					(const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program: compile the kernels in the program object
	// A single program can contain multiple kernels
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

#ifdef ZERO_COPY
	int *a_mapped, *b_mapped, *c_mapped;	// host ptr

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
					LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
					LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
					LIST_SIZE * sizeof(int), NULL, &ret);

	// Map device memory to host address space
	a_mapped = clEnqueueMapBuffer(command_queue, a_mem_obj, 1 /* blocking */,
					CL_MAP_READ | CL_MAP_WRITE, 0, LIST_SIZE * sizeof(int), 0, NULL, NULL, &ret);

	b_mapped = clEnqueueMapBuffer(command_queue, b_mem_obj, 1 /* blocking */,
					CL_MAP_READ | CL_MAP_WRITE, 0, LIST_SIZE * sizeof(int), 0, NULL, NULL, &ret);

	// directly populate 
	for (int i = 0 ; i < LIST_SIZE; i++) {
		a_mapped[i] = i;
		b_mapped[i] = LIST_SIZE - i;
	}

	ret = clEnqueueUnmapMemObject(command_queue, a_mem_obj, a_mapped, 0, NULL, NULL);
	ret = clEnqueueUnmapMemObject(command_queue, b_mem_obj, b_mapped, 0, NULL, NULL);

#else
	int *a = (int*)malloc(sizeof(int)*LIST_SIZE);
	int *b = (int*)malloc(sizeof(int)*LIST_SIZE);
	int *c = (int*)malloc(sizeof(int)*LIST_SIZE);

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
					LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
					LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
					LIST_SIZE * sizeof(int), NULL, &ret);

	for (i = 0; i < LIST_SIZE; i++) {
		a[i] = i;
		b[i] = LIST_SIZE - i;
	}

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
					LIST_SIZE * sizeof(int), a, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
					LIST_SIZE * sizeof(int), b, 0, NULL, NULL);
#endif
	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

	// Execute the OpenCL kernel on the list
	size_t global_item_size = LIST_SIZE; // Process the entire lists
	size_t local_item_size = 64; // Divide work items into groups of 64

	for(int i = 0; i < NRUNS; i++) {
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
						&global_item_size, &local_item_size, 0, NULL, NULL);

	}

	ret = clFlush(command_queue);

#ifdef ZERO_COPY

	c_mapped = clEnqueueMapBuffer(command_queue, c_mem_obj, 1 /* blocking */,
					CL_MAP_READ | CL_MAP_WRITE, 0, LIST_SIZE * sizeof(int), 0, NULL, NULL, &ret);

	// Display the result to the screen
   /**  for(i = 0; i < LIST_SIZE; i++) */
		/** printf("%d + %d = %d\n", i, LIST_SIZE - i, c_mapped[i]); */
	/** printf("Done\n"); */

	ret = clEnqueueUnmapMemObject(command_queue, c_mem_obj, c_mapped, 0, NULL, NULL);

#else
	// Read the memory buffer C on the device to the local variable C
	// Copy back the result from device mem to host mem
	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
					LIST_SIZE * sizeof(int), c, 0, NULL, NULL);

	// Display the result to the screen
   /**  for(i = 0; i < LIST_SIZE; i++) */
		/** printf("%d + %d = %d\n", i, LIST_SIZE - i, c[i]); */
	/** printf("Done\n"); */

	free(a); free(b); free(c);
#endif

	// Clean up
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(source_str);
	return 0;
}


void main() {
	run();
}

