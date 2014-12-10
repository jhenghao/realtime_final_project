/*
     File: NBodySimulationCPU.mm
 Abstract: 
 Utility class for managing cpu bound computes for n-body simulation.
 
  Version: 3.3
 
 Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple
 Inc. ("Apple") in consideration of your agreement to the following
 terms, and your use, installation, modification or redistribution of
 this Apple software constitutes acceptance of these terms.  If you do
 not agree with these terms, please do not use, install, modify or
 redistribute this Apple software.
 
 In consideration of your agreement to abide by the following terms, and
 subject to these terms, Apple grants you a personal, non-exclusive
 license, under Apple's copyrights in this original Apple software (the
 "Apple Software"), to use, reproduce, modify and redistribute the Apple
 Software, with or without modifications, in source and/or binary forms;
 provided that if you redistribute the Apple Software in its entirety and
 without modifications, you must retain this notice and the following
 text and disclaimers in all such redistributions of the Apple Software.
 Neither the name, trademarks, service marks or logos of Apple Inc. may
 be used to endorse or promote products derived from the Apple Software
 without specific prior written permission from Apple.  Except as
 expressly stated in this notice, no other rights or licenses, express or
 implied, are granted by Apple herein, including but not limited to any
 patent rights that may be infringed by your derivative works or by other
 works in which the Apple Software may be incorporated.
 
 The Apple Software is provided by Apple on an "AS IS" basis.  APPLE
 MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION
 THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS
 FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND
 OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
 
 IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL
 OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION,
 MODIFICATION AND/OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED
 AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE),
 STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
 
 Copyright (C) 2014 Apple Inc. All Rights Reserved.
 
 */

#pragma mark -
#pragma mark Private - Headers

#import <cmath>
#import <iostream>
#import <mach/mach_time.h>

#import "CFIFStream.h"

#import "GLMSizes.h"

#import "NBodySimulationRandom.h"
#import "NBodySimulationCPU.h"

#pragma mark -
#pragma mark Private - Utilities

GLint NBody::Simulation::CPU::bind()
{
    GLint err = mpData->bind(mpKernel);
    
    if(err == CL_SUCCESS)
    {
        size_t  sizes[6];
        void   *values[6];
        
        cl_uint indicies[6];
        
        size_t  nWorkGroupCount = (mnMaxIndex - mnMinIndex) / mnUnits;
        GLfloat nTimeStamp      = m_ActiveParams.mnTimeStamp;
        
        values[0] = (void *) &nTimeStamp;
        values[1] = (void *) &m_ActiveParams.mnDamping;
        values[2] = (void *) &m_ActiveParams.mnSoftening;
        values[3] = (void *) &mnBodyCount;
        values[4] = (void *) &nWorkGroupCount;
        values[5] = (void *) &mnMinIndex;
        
        sizes[0] = mnSamples;
        sizes[1] = mnSamples;
        sizes[2] = mnSamples;
        sizes[3] = GLM::Size::kInt;
        sizes[4] = GLM::Size::kInt;
        sizes[5] = GLM::Size::kInt;
        
        indicies[0] = 14;
        indicies[1] = 15;
        indicies[2] = 16;
        indicies[3] = 17;
        indicies[4] = 18;
        indicies[5] = 19;
        
        GLint i;
        
        for(i = 0; i < 6; ++i)
        {
            err = clSetKernelArg(mpKernel, indicies[i], sizes[i], values[i]);
            
            if(err != CL_SUCCESS)
            {
                return err;
            } // if
        } // for
    } // if
    
    return err;
} // bind

GLint NBody::Simulation::CPU::setup(const String& options,
                                    const bool& vectorized,
                                    const bool& threaded)
{
    GLint err = CL_INVALID_VALUE;
    
    CF::IFStreamRef pStream = CF::IFStreamCreate(CFSTR("nbody_cpu"), CFSTR("ocl"));
    
    if(CF::IFStreamIsValid(pStream))
    {
        err = clGetDeviceIDs(nullptr,
                             CL_DEVICE_TYPE_CPU,
                             1,
                             &mpDevice,
                             &mnDeviceCount);
        
        if(err != CL_SUCCESS)
        {
            return err;
        } // if
        
        mpContext = clCreateContext(nullptr,
                                    mnDeviceCount,
                                    &mpDevice,
                                    nullptr,
                                    nullptr,
                                    &err);
        
        if(err != CL_SUCCESS)
        {
            return err;
        } // if
        
        mpQueue = clCreateCommandQueue(mpContext,
                                       mpDevice,
                                       0,
                                       &err);
        
        if(err != CL_SUCCESS)
        {
            return err;
        } // if
        
        size_t returned_size;
        GLuint compute_units;
        
        clGetDeviceInfo(mpDevice,
                        CL_DEVICE_MAX_COMPUTE_UNITS,
                        GLM::Size::kUInt,
                        &compute_units,
                        &returned_size);
        
        mnUnits = threaded ? compute_units : 1;
        
        const char *pBuffer = CF::IFStreamGetBuffer(pStream);
        
        mpProgram = clCreateProgramWithSource(mpContext,
                                              1,
                                              &pBuffer,
                                              nullptr,
                                              &err);
        
        if(err != CL_SUCCESS)
        {
            return err;
        } // if
        
        const char *pOptions = !options.empty() ? options.c_str() : nullptr;
        
        err = clBuildProgram(mpProgram,
                             mnDeviceCount,
                             &mpDevice,
                             pOptions,
                             nullptr,
                             nullptr);
        
        if(err != CL_SUCCESS)
        {
            size_t length = 0;
            
            char info_log[2000];
            
            clGetProgramBuildInfo(mpProgram,
                                  mpDevice,
                                  CL_PROGRAM_BUILD_LOG,
                                  2000,
                                  info_log,
                                  &length);
            
            std::cerr
            << ">> N-body Simulation:"
            << std::endl
            << info_log
            << std::endl;
            
            return err;
        } // if
        
        mpKernel = clCreateKernel(mpProgram,
                                  vectorized ? "IntegrateSystemVectorized" : "IntegrateSystemNonVectorized",
                                  &err);
        
        if(err != CL_SUCCESS)
        {
            return err;
        } // if
        
        err = mpData->acquire(mpContext);
        
        if(err != CL_SUCCESS)
        {
            return err;
        } // if

        err = bind();
        
        CF::IFStreamRelease(pStream);
    } // if
    
    return err;
} // setup

GLint NBody::Simulation::CPU::execute()
{
    GLint err = CL_INVALID_KERNEL;
    
    if(mpKernel != nullptr)
    {
        mpData->update(mpKernel);

        size_t nWorkGroupCount = (mnMaxIndex - mnMinIndex) / mnUnits;
        
        size_t    sizes[2];
        uint32_t  indices[2];
        void      *values[2];
        
        values[0] = &nWorkGroupCount;
        values[1] = &mnMinIndex;
        
        sizes[0] = GLM::Size::kInt;
        sizes[1] = GLM::Size::kInt;
        
        indices[0] = 18;
        indices[1] = 19;
        
        GLint i;
        
        for(i = 0; i < 2; ++i)
        {
            err = clSetKernelArg(mpKernel,
                                 indices[i],
                                 sizes[i],
                                 values[i]);
            
            if(err != CL_SUCCESS)
            {
                return err;
            } // if
        } // for
        
        if(mpQueue != nullptr)
        {
            size_t global_dim[2];
            size_t local_dim[2];
            
            local_dim[0]  = 1;
            local_dim[1]  = 1;
            
            global_dim[0] = mnUnits;
            global_dim[1] = 1;
            
            err = clEnqueueNDRangeKernel(mpQueue,
                                         mpKernel,
                                         2,
                                         nullptr,
                                         global_dim,
                                         local_dim,
                                         0,
                                         nullptr,
                                         nullptr);
            
            if(err != CL_SUCCESS)
            {
                return err;
            } // if
            
            err = clFinish(mpQueue);
            
            if(err != CL_SUCCESS)
            {
                return err;
            } // if
        } // if
    } // if
    
    return err;
} // execute

GLint NBody::Simulation::CPU::restart()
{
    mpData->reset(m_ActiveParams);
    
    return bind();
} // restart

#pragma mark -
#pragma mark Public - Constructor

NBody::Simulation::CPU::CPU(const size_t& nbodies,
                            const NBody::Simulation::Params& params,
                            const bool& vectorized,
                            const bool& threaded)
: NBody::Simulation::Base(nbodies, params)
{
    mbVectorized = vectorized;
    mbThreaded   = threaded;
    mbTerminated = false;
    mnUnits      = 0;
    mpDevice     = nullptr;
    mpQueue      = nullptr;
    mpContext    = nullptr;
    mpProgram    = nullptr;
    mpKernel     = nullptr;
    mpData       = new NBody::Simulation::Data::Mediator(mnBodyCount);
} // Constructor

#pragma mark -
#pragma mark Public - Destructor

NBody::Simulation::CPU::~CPU()
{
    stop();
    
    terminate();
} // Destructor

#pragma mark -
#pragma mark Public - Utilities

void NBody::Simulation::CPU::initialize(const NBody::Simulation::String& options)
{
    if(!mbTerminated)
    {
        GLint err = setup(options, mbVectorized, mbThreaded);
        
        mbAcquired = err == CL_SUCCESS;
        
        if(!mbAcquired)
        {
            std::cerr
            << ">> N-body Simulation["
            << err
            << "]: Failed setting up cpu compute device!"
            << std::endl;
        } // if
    } // if
} // initialize

GLint NBody::Simulation::CPU::reset()
{
    GLint err = restart();
    
    if(err != 0)
    {
        std::cerr
        << ">> N-body Simulation["
        << err
        << "]: Failed resetting devices!"
        << std::endl;
    } // if
    
    return err;
} // reset

bool NBody::Simulation::CPU::step()
{
    if(!isPaused() || !isStopped())
    {
        GLint err = execute();
        
        if((err != 0) && (!mbTerminated))
        {
            std::cerr
            << ">> N-body Simulation["
            << err
            << "]: Failed executing vectorized & threaded kernel!"
            << std::endl;
        } // if
        
        if(mbIsUpdated)
        {
            setData(mpData->position());
        } // if
        
        mpData->swap();
    } // if
    
    return false;
} // step

void NBody::Simulation::CPU::fibonacci_step(int number_1, int number_2, int count)
{
    uint64_t start = mach_absolute_time();
    
    int temp_1 = number_1;
    int temp_2 = number_2;
    int result = 0;
    
    for (int idx = 0; idx < count; ++idx)
    {
        result = temp_1 + temp_2;
        temp_1 = temp_2;
        temp_2 = result;
    }
    
    uint64_t end = mach_absolute_time();
    uint64_t elapsed = end - start;
    Nanoseconds elapsedNano = AbsoluteToNanoseconds(*(AbsoluteTime *) &elapsed);
    
    std::cout << "fibonacci exec time (ns) on CPU: " << (*(uint64_t *) &elapsedNano) << std::endl;
    //printf("fibonacci(%d): %d\n", count, result);
}

void NBody::Simulation::CPU::terminate()
{
    if(!mbTerminated)
    {
        if(mpQueue != nullptr)
        {
            clFinish(mpQueue);
        } // if
        
        if(mpData != nullptr)
        {
            delete mpData;
            
            mpData = nullptr;
        } // if
        
        if(mpQueue != nullptr)
        {
            clReleaseCommandQueue(mpQueue);
            
            mpQueue = nullptr;
        } // if
        
        if(mpKernel != nullptr)
        {
            clReleaseKernel(mpKernel);
            
            mpKernel = nullptr;
        } // if
        
        if(mpProgram != nullptr)
        {
            clReleaseProgram(mpProgram);
            
            mpProgram = nullptr;
        } // if
        
        if(mpContext != nullptr)
        {
            clReleaseContext(mpContext);
            
            mpContext = nullptr;
        } // if
        
        mbTerminated = true;
    } // if
} // terminate

#pragma mark -
#pragma mark Public - Accessors

GLint NBody::Simulation::CPU::positionInRange(GLfloat *pDst)
{
    return mpData->positionInRange(mnMinIndex, mnMaxIndex, pDst);
} // positionInRange

GLint NBody::Simulation::CPU::position(GLfloat *pDst)
{
    return mpData->position(mnMaxIndex, pDst);
} // position

GLint NBody::Simulation::CPU::setPosition(const GLfloat * const pSrc)
{
    return mpData->setPosition(pSrc);
} // setPosition

GLint NBody::Simulation::CPU::velocity(GLfloat *pDst)
{
    return mpData->velocity(pDst);
} // velocity

GLint NBody::Simulation::CPU::setVelocity(const GLfloat * const pSrc)
{
    return mpData->setVelocity(pSrc);
} // setVelocity

bool NBody::Simulation::CPU::runFibonacciTask(int number_1, int number_2, int count)
{
    if (TaskType::NONE != task_) return false;
    
    number_1_ = number_1;
    number_2_ = number_2;
    count_ = count;
    task_ = TaskType::FIBONACCI;
    return true;
}

bool NBody::Simulation::CPU::runNBodyTask ()
{
    if (TaskType::NONE != task_) return false;
    
    task_ = TaskType::NBODY;
    return true;
}
