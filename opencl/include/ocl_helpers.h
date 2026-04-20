/*
 * ocl_helpers.h
 * OpenCL boilerplate helpers: context, queue, program, etc.
 */
#ifndef OCL_HELPERS_H
#define OCL_HELPERS_H

#include <CL/cl.h>

/* Print error and exit on failure */
void checkErr(cl_int err, const char *name);

/* Discover platform and device, create context */
cl_context createContext(
    cl_platform_id *platform_out,
    cl_device_id   *device_out);

/* Create a profiling-enabled command queue */
cl_command_queue createCommandQueue(
    cl_context    context,
    cl_device_id  device);

/* Build a program from a .cl source file */
cl_program createProgram(
    cl_context   context,
    cl_device_id device,
    const char  *filename);

/* Print platform and device info to stdout */
void printDeviceInfo(
    cl_platform_id platform,
    cl_device_id   device);

#endif /* OCL_HELPERS_H */
