/*
     File: NBodySimulationMediator.mm
 Abstract: 
 A mediator object for managing cpu and gpu bound simulators, along with their labeled-buttons.
 
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

#import <iostream>

#import <mach/mach_time.h>
#import <OpenCL/OpenCL.h>
#import <OpenGL/gl.h>

#import "GLMSizes.h"

#import "NBodySimulationMediator.h"

static const GLuint kNBodyMaxDeviceCount = 128;

// Get the number of coumpute device counts
static GLuint NBodyGetComputeDeviceCount(const GLint& type)
{
    cl_device_id ids[kNBodyMaxDeviceCount] = {0};
    
    GLuint count = -1;
    
    GLint err = clGetDeviceIDs(nullptr, type, kNBodyMaxDeviceCount, ids, &count);
    
    if(err != CL_SUCCESS)
    {
        std::cerr
        << ">> ERROR: NBody Simulation Mediator - Failed acquiring maximum device count!"
        << std::endl;
    } // if
    
    return count;
} // NBodyGetComputeDeviceCount

// Set the current active n-body parameters
void NBody::Simulation::Mediator::setParams(const NBody::Simulation::Params& rParams)
{
    std::memset(&m_Params, 0x0, sizeof(NBody::Simulation::Params));
    
    m_Params = rParams;
} // setParams

// Set the defaults for simulator compute
void NBody::Simulation::Mediator::setCompute(const bool& bGPUOnly)
{
    mnGPUs = NBodyGetComputeDeviceCount(CL_DEVICE_TYPE_GPU);
    
    if(!bGPUOnly)
    {
        mbCPUs = NBodyGetComputeDeviceCount(CL_DEVICE_TYPE_CPU) > 0;
    } // if
    
    mnActive = (mbCPUs)
    ? NBody::Simulation::eComputeCPUSingle
    : NBody::Simulation::eComputeGPUPrimary;
} // setCompute

// Initialize all instance variables to their default values
void NBody::Simulation::Mediator::setDefaults(const size_t& nBodies)
{
    
    mnBodies = nBodies;
    mnSize   = 4 * mnBodies * GLM::Size::kFloat;
    
    mnCount    = 0;
    mnGPUs     = 0;
    mbCPUs     = false;
    mpActive   = nullptr;
    mpPosition = nullptr;
    
    mpSimulators[NBody::Simulation::eComputeCPUSingle]    = nullptr;
    mpSimulators[NBody::Simulation::eComputeCPUMulti]     = nullptr;
    mpSimulators[NBody::Simulation::eComputeGPUPrimary]   = nullptr;
    mpSimulators[NBody::Simulation::eComputeGPUSecondary] = nullptr;
} // setDefaults

// Acquire all simulators
void NBody::Simulation::Mediator::acquire(const NBody::Simulation::Params& rParams, void * p_engine)
{
    setParams(rParams);
    
    if(mnGPUs > 0)
    {
        mpSimulators[NBody::Simulation::eComputeGPUPrimary]
        = new NBody::Simulation::Facade(NBody::Simulation::eComputeGPUPrimary, mnBodies, m_Params, p_engine);
        
        if(mpSimulators[NBody::Simulation::eComputeGPUPrimary] != nullptr)
        {
            mnCount++;
        } // if
    } // if
    
    if(mnGPUs > 1)
    {
        mpSimulators[NBody::Simulation::eComputeGPUSecondary]
        = new NBody::Simulation::Facade(NBody::Simulation::eComputeGPUSecondary, mnBodies, m_Params, p_engine);

        if(mpSimulators[NBody::Simulation::eComputeGPUSecondary] != nullptr)
        {
            mnCount++;
        } // if
    } // if
    
    if(mbCPUs)
    {
        mpSimulators[NBody::Simulation::eComputeCPUSingle]
        = new NBody::Simulation::Facade(NBody::Simulation::eComputeCPUSingle, mnBodies, m_Params, p_engine);
        
        if(mpSimulators[NBody::Simulation::eComputeCPUSingle] != nullptr)
        {
            mnCount++;
        } // if
        
        mpSimulators[NBody::Simulation::eComputeCPUMulti]
        = new NBody::Simulation::Facade(NBody::Simulation::eComputeCPUMulti, mnBodies, m_Params, p_engine);
        
        if(mpSimulators[NBody::Simulation::eComputeCPUMulti] != nullptr)
        {
            mnCount++;
        } // if
    } // if
    
    mnActive = (mbCPUs) ? NBody::Simulation::eComputeCPUSingle : NBody::Simulation::eComputeGPUPrimary;
    mpActive = mpSimulators[mnActive];
} // acquire

// Construct a mediator object for GPUs, or CPU and CPUs
NBody::Simulation::Mediator::Mediator(void * p_engine,
                                      const NBody::Simulation::Params& rParams,
                                      const bool& bGPUOnly,
                                      const GLuint& nCount)
{
    p_cur_nbody_simulator_ = NULL;
    p_latest_render_data_ = NULL;
    
    pthread_mutexattr_init(&render_attrib_);
    pthread_mutexattr_settype(&render_attrib_, PTHREAD_MUTEX_RECURSIVE);
    
    pthread_mutex_init(&render_lock_, &render_attrib_);
    
    setDefaults(nCount);
    setCompute(bGPUOnly);
    
    acquire(rParams, p_engine);
} // Constructor

// Delete alll simulators
NBody::Simulation::Mediator::~Mediator()
{
    GLuint i;
    
    for(i = 0; i < NBody::Simulation::eComputeMax; ++i)
    {
        if(mpSimulators[i] != nullptr)
        {
            delete mpSimulators[i];
            
            mpSimulators[i] = nullptr;
        } // if
    } // for
    
    mnCount = 0;
    mnGPUs  = 0;
    mbCPUs  = false;
    
    std::memset(&m_Params, 0x0, sizeof(NBody::Simulation::Params));
} // Destructor

// Is single core cpu simulator active?
const bool NBody::Simulation::Mediator::isCPUSingleCore() const
{
    return mpActive->isCPUSingleCore();
} // isCPUSingleCore

// Is multi-core cpu simulator active?
const bool NBody::Simulation::Mediator::isCPUMultiCore() const
{
    return mpActive->isCPUMultiCore();
} // isCPUMultiCore

// Is primary gpu simulator active?
const bool NBody::Simulation::Mediator::isGPUPrimary() const
{
    return mpActive->isGPUPrimary();
} // isGPUPrimary

// Is secondary (or offline) gpu simulator active?
const bool NBody::Simulation::Mediator::isGPUSecondary() const
{
    return mpActive->isGPUSecondary();
} // isGPUSecondary

// Check to see if position was acquired
const bool NBody::Simulation::Mediator::hasPosition() const
{
    return mpPosition != nullptr;
} // hasPosition

void NBody::Simulation::Mediator::schedule()
{
    static uint64_t last_time_stamp = 0;
    static bool first = true;
    
    if (first)
    {
        first = false;
        last_time_stamp = mach_absolute_time();
        return;
    }
    
    if (NULL != p_cur_nbody_simulator_)
    {
        if (!p_cur_nbody_simulator_->hasTask())
        {
            pthread_mutex_lock(&render_lock_);
            p_latest_render_data_ = p_cur_nbody_simulator_;
            pthread_mutex_unlock(&render_lock_);
        }
    }
    
    uint64_t cur_time_stamp = mach_absolute_time();
    
    uint64_t elapsed = cur_time_stamp - last_time_stamp;
    last_time_stamp = cur_time_stamp;
    
    Nanoseconds elapsedNano = AbsoluteToNanoseconds(*(AbsoluteTime *) &elapsed);
    //std::cout << "time: " << (*(uint64_t *) &elapsedNano) << std::endl;
    
    static uint64_t total_time_elapsed = 0;
    total_time_elapsed += (*(uint64_t *) &elapsedNano);
    
    // parameters
    const int FIBONACCI_PERIOD = 50000000;
    const int NBODY_PERIOD = FIBONACCI_PERIOD;
    
    if (total_time_elapsed < FIBONACCI_PERIOD) return;
    
    total_time_elapsed -= FIBONACCI_PERIOD;
    
    Facade *cpu = mpSimulators[NBody::Simulation::eComputeCPUSingle];
    Facade *gpu = mpSimulators[NBody::Simulation::eComputeGPUPrimary];
    
    const int FIBONACCI_COUNT = 10;

/*
    static int measure_count = 0;
    if (!gpu->hasTask())
    {
        if (measure_count < 200)
        {
            measure_count += 1;
            gpu->fibonacci_step(0, 1, FIBONACCI_COUNT);
        }
    }
*/
    const int PERIOD_MAX = 500;
    static int period_count = 0;
    static int fibonacci_count = 0;
    static int nbody_count = 0;
    
    if (period_count >= PERIOD_MAX)
    {
        std::cout << "fibonacci: " << fibonacci_count << " / " << PERIOD_MAX << std::endl;
            
        std::cout << "nbody: " << nbody_count << " / " << PERIOD_MAX << std::endl;
        return;
    }
    ++period_count;
/*
    if (!gpu->hasTask())
    {
        // fibonacci task is assigned to GPU
        gpu->fibonacci_step(0, 1, FIBONACCI_COUNT);
        ++fibonacci_count;
        
        if (cpu->hasTask())
        {
            //std::cout << "cpu is occupied" << std::endl;
        }
        else
        {
            cpu->nbody_step();
            ++nbody_count;
            p_cur_nbody_simulator_ = cpu;
        }
    }
    else
    {
        if (cpu->hasTask())
        {
            //std::cout << "both cpu and gpu are occupied" << std::endl;
        }
        else
        {
            cpu->fibonacci_step(0, 1, FIBONACCI_COUNT);
            ++fibonacci_count;
        }
    }
*/


    if (!gpu->hasTask())
    {
        // fibonacci task is assigned to GPU
        gpu->nbody_step();
        ++nbody_count;
        p_cur_nbody_simulator_ = gpu;
        
        if (cpu->hasTask())
        {
            std::cout << "cpu is occupied" << std::endl;
        }
        else
        {
            cpu->fibonacci_step(0, 1, FIBONACCI_COUNT);
            ++fibonacci_count;
        }
    }
    else
    {
        if (cpu->hasTask())
        {
            std::cout << "both cpu and gpu are occupied" << std::endl;
        }
        else
        {
            cpu->fibonacci_step(0, 1, FIBONACCI_COUNT);
            ++fibonacci_count;
        }
    }

}

// Get the total number of simulators
const GLuint NBody::Simulation::Mediator::getCount() const
{
    return mnCount;
} // getCount

// Get the relative performance number
const GLdouble NBody::Simulation::Mediator::performance() const
{
    return (mpActive != nullptr) ? mpActive->performance() : 0.0f;
} // performance

// Get the updates performance number
const GLdouble NBody::Simulation::Mediator::updates() const
{
    return (mpActive != nullptr) ? mpActive->updates() : 0.0f;
} // updates

// Pause the current active simulator
void NBody::Simulation::Mediator::pause()
{
    if(mpActive != nullptr)
    {
        mpActive->pause();
    } // if
} // pause

// unpause the current active simulator
void NBody::Simulation::Mediator::unpause()
{
    if(mpActive != nullptr)
    {
        mpActive->unpause();
    } // if
} // unpause

void NBody::Simulation::Mediator::nbody_step()
{
    if (mpActive != nullptr)
    {
        mpActive->nbody_step();
    }
}

void NBody::Simulation::Mediator::fibonacci_step(int number_1, int number_2, int count)
{
    if (mpActive != nullptr)
    {
        mpActive->fibonacci_step(number_1, number_2, count);
    }
}

// Set the button for the current simulator object
void NBody::Simulation::Mediator::button(const bool& selected,
                                         const CGPoint& position,
                                         const CGRect& bounds)
{
    mpActive->button(selected, position, bounds);
} // button

// Select the current simulator to use
void NBody::Simulation::Mediator::select(const NBody::Simulation::Types& type)
{
    if(mpSimulators[type] != nullptr)
    {
        Facade *p_old_simulator = mpSimulators[mnActive];
        
        std::cout << "size: " << p_old_simulator->size() << std::endl;
        std::cout << "length: " << p_old_simulator->length() << std::endl;
        
        mnActive = type;
        mpActive = mpSimulators[mnActive];
        
        GLfloat temp[p_old_simulator->length()];
        
        p_old_simulator->position(temp);
        mpActive->setPosition(temp);
        
        p_old_simulator->velocity(temp);
        mpActive->setVelocity(temp);
        
        
        std::cout
        << ">> N-body Simulation: Using \""
        << mpActive->label()
        << "\" simulator with ["
        << mnBodies
        << "] bodies."
        << std::endl;
    } // if
    else
    {
        std::cout
        << ">> ERROR: N-body Simulation: Requested simulator is nullptr!"
        << std::endl;
    } // else
} // select

// Select the current simulator to use
void NBody::Simulation::Mediator::select(const GLuint& index)
{
    NBody::Simulation::Types type = NBody::Simulation::eComputeMax;
    
    if(mbCPUs)
    {
        switch(index)
        {
            case 0:
                type = NBody::Simulation::eComputeCPUSingle;
                break;
                
            case 1:
                type = NBody::Simulation::eComputeCPUMulti;
                break;
                
            case 3:
                type = NBody::Simulation::eComputeGPUSecondary;
                break;
                
            case 2:
            default:
                type = NBody::Simulation::eComputeGPUPrimary;
                break;
        } // switch
    } // if
    else
    {
        switch(index)
        {
            case 1:
                type = NBody::Simulation::eComputeGPUSecondary;
                break;
                
            case 0:
                type = NBody::Simulation::eComputeGPUPrimary;
                break;
        } // switch
    } // else

    select(type);
} // select

// Get position data
const GLfloat* NBody::Simulation::Mediator::position() const
{
    return mpPosition;
} // position

// Get the current simulator
NBody::Simulation::Facade* NBody::Simulation::Mediator::simulator()
{
    return mpActive;
} // simulator

// void update position data
void NBody::Simulation::Mediator::update()
{
    //GLfloat *pPosition = mpActive->data();
    
    if (NULL == p_latest_render_data_) return;
    pthread_mutex_lock(&render_lock_);
    GLfloat *pPosition = p_latest_render_data_->data();
    pthread_mutex_unlock(&render_lock_);
    
    if(pPosition != nullptr)
    {
        if(mpPosition != nullptr)
        {
            free(mpPosition);
        } // if
        
        mpPosition = pPosition;
    } // if
} // update

// Reset all the gpu bound simulators
void NBody::Simulation::Mediator::reset()
{
    if(mpActive != nullptr)
    {
        if(mpPosition != nullptr)
        {
            free(mpPosition);
            
            mpPosition = nullptr;
            
            free(mpActive->data());
        } // if
        
        mpActive->resetParams(m_Params);

mpSimulators[NBody::Simulation::eComputeCPUSingle]->resetParams(m_Params);
mpSimulators[NBody::Simulation::eComputeGPUPrimary]->resetParams(m_Params);
    
        if(    (mnActive == NBody::Simulation::eComputeGPUPrimary)
           &&  (mpSimulators[NBody::Simulation::eComputeGPUSecondary] != nullptr))
        {
            mpSimulators[NBody::Simulation::eComputeGPUPrimary]->invalidate(true);
            mpSimulators[NBody::Simulation::eComputeGPUSecondary]->invalidate(false);
        } // if
        else if(    (mnActive == NBody::Simulation::eComputeGPUSecondary)
                &&  (mpSimulators[NBody::Simulation::eComputeGPUPrimary] != nullptr))
        {
            mpSimulators[NBody::Simulation::eComputeGPUPrimary]->invalidate(false);
            mpSimulators[NBody::Simulation::eComputeGPUSecondary]->invalidate(true);
        } // else if
        else if (mnActive == NBody::Simulation::eComputeGPUPrimary)
        {
            mpSimulators[NBody::Simulation::eComputeGPUPrimary]->invalidate(true);
        } // else if
        
        //mpActive->unpause();
    } // if
} // NBodyResetSimulators
