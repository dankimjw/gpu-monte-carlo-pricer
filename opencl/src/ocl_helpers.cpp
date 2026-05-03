/*
 * ocl_helpers.cpp
 * OpenCL boilerplate: platform/device discovery, context,
 * command queue, and program compilation helpers.
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <CL/cl.h>
#include "../include/ocl_helpers.h"

/* ---------------------------------------------------- */
void checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
                  << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

/* ---------------------------------------------------- */
cl_context createContext(
    cl_platform_id *platform_out,
    cl_device_id   *device_out)
{
    cl_int  errNum;
    cl_uint numPlatforms;

    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(
        (errNum != CL_SUCCESS || numPlatforms == 0)
            ? -1 : CL_SUCCESS,
        "clGetPlatformIDs");

    cl_platform_id *platforms = new cl_platform_id[numPlatforms];
    errNum = clGetPlatformIDs(numPlatforms, platforms, NULL);
    checkErr(errNum, "clGetPlatformIDs(get)");

    /* Find a platform with a GPU device */
    cl_device_id device = 0;
    cl_platform_id platform = 0;
    for (cl_uint i = 0; i < numPlatforms; i++) {
        cl_uint numDevices = 0;
        errNum = clGetDeviceIDs(
            platforms[i], CL_DEVICE_TYPE_GPU,
            0, NULL, &numDevices);
        if (errNum == CL_SUCCESS && numDevices > 0) {
            errNum = clGetDeviceIDs(
                platforms[i], CL_DEVICE_TYPE_GPU,
                1, &device, NULL);
            checkErr(errNum, "clGetDeviceIDs");
            platform = platforms[i];
            break;
        }
    }
    delete[] platforms;

    if (device == 0) {
        std::cerr << "No GPU device found." << std::endl;
        exit(EXIT_FAILURE);
    }

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    cl_context context = clCreateContext(
        props, 1, &device, NULL, NULL, &errNum);
    checkErr(errNum, "clCreateContext");

    *platform_out = platform;
    *device_out   = device;
    return context;
}

/* ---------------------------------------------------- */
cl_command_queue createCommandQueue(
    cl_context   context,
    cl_device_id device)
{
    cl_int errNum;
    cl_command_queue queue = clCreateCommandQueue(
        context, device,
        CL_QUEUE_PROFILING_ENABLE,
        &errNum);
    checkErr(errNum, "clCreateCommandQueue");
    return queue;
}

/* ---------------------------------------------------- */
cl_program createProgram(
    cl_context   context,
    cl_device_id device,
    const char  *filename)
{
    cl_int errNum;

    std::ifstream srcFile(filename);
    if (!srcFile.is_open()) {
        std::cerr << "Cannot open kernel file: "
                  << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::ostringstream oss;
    oss << srcFile.rdbuf();
    std::string srcStr = oss.str();
    const char *src = srcStr.c_str();
    size_t len = srcStr.length();

    cl_program program = clCreateProgramWithSource(
        context, 1, &src, &len, &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    errNum = clBuildProgram(
        program, 1, &device, "-I.", NULL, NULL);
    if (errNum != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(
            program, device,
            CL_PROGRAM_BUILD_LOG,
            sizeof(log), log, NULL);
        std::cerr << "Build error:\n" << log << std::endl;
        checkErr(errNum, "clBuildProgram");
    }
    return program;
}

/* ---------------------------------------------------- */
void printDeviceInfo(
    cl_platform_id platform,
    cl_device_id   device)
{
    char buf[256];

    clGetPlatformInfo(
        platform, CL_PLATFORM_NAME,
        sizeof(buf), buf, NULL);
    std::cout << "Platform:       " << buf << std::endl;

    clGetDeviceInfo(
        device, CL_DEVICE_NAME,
        sizeof(buf), buf, NULL);
    std::cout << "Device:         " << buf << std::endl;

    cl_uint compute_units = 0;
    clGetDeviceInfo(
        device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(compute_units), &compute_units, NULL);
    std::cout << "Compute units:  "
              << compute_units << std::endl;

    cl_ulong global_mem = 0;
    clGetDeviceInfo(
        device, CL_DEVICE_GLOBAL_MEM_SIZE,
        sizeof(global_mem), &global_mem, NULL);
    std::cout << "Global memory:  "
              << (global_mem / (1024 * 1024)) << " MB"
              << std::endl;

    cl_ulong local_mem = 0;
    clGetDeviceInfo(
        device, CL_DEVICE_LOCAL_MEM_SIZE,
        sizeof(local_mem), &local_mem, NULL);
    std::cout << "Local memory:   "
              << (local_mem / 1024) << " KB"
              << std::endl;

    size_t max_wg = 0;
    clGetDeviceInfo(
        device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(max_wg), &max_wg, NULL);
    std::cout << "Max work-group: "
              << max_wg << std::endl;
}
