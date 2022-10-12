//From https://github.com/mnicely/nvml_examples

/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#ifndef NVMLCLASS_H_
#define NVMLCLASS_H_ 1

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <nvml.h>

int constexpr size_of_vector { 100000 };
int constexpr nvml_device_name_buffer_size { 100 };
double constexpr convert_mJ { 1000000000 };
double constexpr convert_ns_S { 1000000000};

// *************** FOR ERROR CHECKING *******************
#ifndef NVML_RT_CALL
#define NVML_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<nvmlReturn_t>( call );                                                               \
        if ( status != NVML_SUCCESS )                                                                                  \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed "                                      \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     nvmlErrorString( status ),                                                                        \
                     status );                                                                                         \
    }
#endif  // NVML_RT_CALL
// *************** FOR ERROR CHECKING *******************

class nvmlClass {
  public:
    nvmlClass( int const &deviceID, std::string  &filename, std::string &stats_filename, std::string type ) :
               time_steps_ {}, stat_pts_time_steps_ {}, 
               filename_ { filename }, stats_filename_ { stats_filename }, outfile_ {},
               device_ {}, loop_ { false } , type_ { type } {

        char name[nvml_device_name_buffer_size];

        // Initialize NVML library
        NVML_RT_CALL( nvmlInit( ) );

        // Query device handle
        NVML_RT_CALL( nvmlDeviceGetHandleByIndex( deviceID, &device_ ) );

        // Query device name
        NVML_RT_CALL( nvmlDeviceGetName( device_, name, nvml_device_name_buffer_size ) );

        // Reserve memory for data
        time_steps_.reserve( size_of_vector );

        // Reserve memory for significant pts
        stat_pts_time_steps_.reserve( size_of_vector );

        // Open output file
        outfile_.open( filename_, std::ios::out );

        // Open stats file
        stats_file_.open( stats_filename_, std::ios::out );

        // Print header
        printHeader( );
    }

    ~nvmlClass( ) {

        NVML_RT_CALL( nvmlShutdown( ) );
    }

    void getStats( ) {

        stats device_stats {};
        loop_ = true;

        // Collect power information
        while ( loop_ ) {
            device_stats.timestamp = std::chrono::high_resolution_clock::now( ).time_since_epoch( ).count( );
            NVML_RT_CALL( nvmlDeviceGetTemperature( device_, NVML_TEMPERATURE_GPU, &device_stats.temperature ) );
            NVML_RT_CALL( nvmlDeviceGetPowerUsage( device_, &device_stats.powerUsage ) );
            NVML_RT_CALL( nvmlDeviceGetEnforcedPowerLimit( device_, &device_stats.powerLimit ) );
            NVML_RT_CALL( nvmlDeviceGetClockInfo( device_, NVML_CLOCK_GRAPHICS, &device_stats.graphicsClock));
            NVML_RT_CALL( nvmlDeviceGetClockInfo( device_, NVML_CLOCK_MEM, &device_stats.memClock));
            NVML_RT_CALL( nvmlDeviceGetClockInfo( device_, NVML_CLOCK_SM, &device_stats.smClock));

            time_steps_.push_back( device_stats );

            std::this_thread::sleep_for( std::chrono::milliseconds(5));
        }

        // Collect information for a short period of time (cooldown) after loop_ is flagged to be false
        //1000 * ~5 ms = 5s
        for (int i = 0; i < 1000; i++) {

          device_stats.timestamp = std::chrono::high_resolution_clock::now( ).time_since_epoch( ).count( );
          NVML_RT_CALL( nvmlDeviceGetTemperature( device_, NVML_TEMPERATURE_GPU, &device_stats.temperature ) );
          NVML_RT_CALL( nvmlDeviceGetPowerUsage( device_, &device_stats.powerUsage ) );
          NVML_RT_CALL( nvmlDeviceGetEnforcedPowerLimit( device_, &device_stats.powerLimit ) );
          NVML_RT_CALL( nvmlDeviceGetClockInfo( device_, NVML_CLOCK_GRAPHICS, &device_stats.graphicsClock));
          NVML_RT_CALL( nvmlDeviceGetClockInfo( device_, NVML_CLOCK_MEM, &device_stats.memClock));
          NVML_RT_CALL( nvmlDeviceGetClockInfo( device_, NVML_CLOCK_SM, &device_stats.smClock));

          // Push information into stats vector
          time_steps_.push_back( device_stats );

          // Sleep for a short period of time
          std::this_thread::sleep_for( std::chrono::milliseconds(5));
        }

        writeData();
    }

    void killThread( ) {
        // Set loop to false to exit while loop
        loop_ = false;
    }

    void new_experiment(std::string  &filename, std::string &stats_filename, std::string &type) {
      
      filename_ = filename;
      stats_filename_ = stats_filename;
      type_ = type;

      // Open output file
      outfile_.open( filename_, std::ios::out );

      // Open stats output file
      stats_file_.open( stats_filename);

      // Print header
      printHeader( );
    }

    void ping_start() {

      // Retrieve empty samples for 5s
      std::this_thread::sleep_for( std::chrono::seconds(5));

      ping_point();
    }

    void ping_point() {
      
      stat_pts device_stats {};

      NVML_RT_CALL( nvmlDeviceGetPowerUsage( device_, &device_stats.powerUsage ) );
      device_stats.timestamp = std::chrono::high_resolution_clock::now( ).time_since_epoch( ).count( );

      stat_pts_time_steps_.push_back(device_stats);
    }

  private:
    typedef struct _stats {
        std::time_t        timestamp;
        uint               temperature;
        uint               powerUsage;
        uint               powerLimit;
        uint               graphicsClock;
        uint               smClock;
        uint               memClock;
    } stats;

    typedef struct _stat_pts {
        time_t             timestamp;
        uint               powerUsage;
    } stat_pts;

    std::vector<std::string> names_ = { "timestep",
                                        "timestamp",
                                        "temperature_gpu",
                                        "type",
                                        "power_draw_mW",
                                        "power_limit_mW",
                                        "g_clock_freq_mhz",
                                        "mem_clock_freq_mhz",
                                        "sm_clock_freq_mhz"};

    std::vector<stats>        time_steps_;
    std::vector<stat_pts>     stat_pts_time_steps_;
    std::string               filename_;
    std::string               stats_filename_;
    std::string               type_;
    std::ofstream             outfile_;
    std::ofstream             stats_file_;
    nvmlDevice_t              device_;
    bool                      loop_;

    void printHeader( ) {

        // Print header
        for ( int i = 0; i < ( static_cast<int>( names_.size( ) ) - 1 ); i++ )
            outfile_ << names_[i] << ",";
        // Leave off the last comma
        outfile_ << names_[static_cast<int>( names_.size( ) ) - 1];
        outfile_ << "\n";
    }

    void writeData( ) {

        // Caclulated max stats
        uint max_power = 0;
        uint max_step  = 0;
        time_t max_timestamp = 0;

        // Calculated min stats
        uint min_power = UINT_MAX;
        uint min_step  = 0;
        time_t min_timestamp = 0;

        // Energy stats
        int current_stat_pt = 0;
        time_t total_time = 0;
        double total_energy = 0;
        double step_energy = 0;

        // Vector sizes
        int total_time_steps = static_cast<int>( time_steps_.size( ));
        int total_stat_pts = static_cast<int> ( stat_pts_time_steps_.size( ));

        // Analyze stats
        // Note: We ignore the very last step of time_steps_ so integration works properly
        for ( int i = 0; i < (total_time_steps - 1 ); i++ ) {

            // Current information
            time_t current_timestamp = time_steps_[i].timestamp;
            uint   current_power = time_steps_[i].powerUsage;

            // Next information
            time_t next_timestamp = time_steps_[i+1].timestamp;
            uint   next_power = time_steps_[i+1].powerUsage;

            outfile_ << i << ","
                     << current_timestamp << ","
                     << time_steps_[i].temperature << ","
                     << type_ << ","
                     << current_power << ","  // mW
                     << time_steps_[i].powerLimit << ","  // mW
                     << time_steps_[i].graphicsClock << "," //MHz
                     << time_steps_[i].memClock << "," //MHz
                     << time_steps_[i].smClock << "\n"; //MHz            

            // Calculate energy
            double h_time = (double) next_timestamp - current_timestamp;
            double temp_energy = 0.5 * (current_power + next_power) * h_time;

            // Update energy totals
            step_energy += temp_energy;
            total_energy += temp_energy;

            // Update timestep and report information from timestep
            if ((next_timestamp > stat_pts_time_steps_[current_stat_pt].timestamp) && (current_stat_pt < total_stat_pts)) {
              
              stats_file_ << "-------------------------------------------------------------------------------------\n"; 

              if (current_stat_pt > 0) {
                stats_file_ << "Step from " << current_stat_pt - 1 << " to " << current_stat_pt
                          << " energy (mJ): " << step_energy / convert_mJ << "\n";

                stats_file_ << "Step from " << current_stat_pt - 1 << " to " << current_stat_pt
                            << " time elapsed: " 
                            << (stat_pts_time_steps_[current_stat_pt].timestamp - stat_pts_time_steps_[current_stat_pt - 1].timestamp) / convert_ns_S
                            << " seconds\n";
              } else {
                stats_file_ << "Step from beginning to " << current_stat_pt
                            << " energy (mJ): " << step_energy / convert_mJ << "\n";
              
                stats_file_ << "Step from beginning to " << current_stat_pt
                            << " time elapsed: " 
                            << (stat_pts_time_steps_[current_stat_pt].timestamp - time_steps_[0].timestamp) / convert_ns_S
                            << " seconds\n";
              }

              step_energy = 0;
              current_stat_pt++;
            }

            // Max and min
            if (max_power < current_power) {
              max_power = current_power;
              max_step = i;
              max_timestamp = current_timestamp;
            }

            if (min_power > current_power) {
              min_power = current_power;
              min_step = i;
              min_timestamp = current_timestamp;
            }
        }

        //Last point to end
        stats_file_ << "-------------------------------------------------------------------------------------\n";

        stats_file_ << "Step from " << current_stat_pt - 1 << " to end"
                    << " energy (mJ): " << step_energy / convert_mJ << "\n";

        stats_file_ << "Step from " << current_stat_pt - 1 << " to end"
                            << " time elapsed: " 
                            << (time_steps_[total_time_steps - 1].timestamp - stat_pts_time_steps_[current_stat_pt - 1].timestamp) / convert_ns_S
                            << " seconds\n";

        // Print energy information
        stats_file_ << "-------------------------------------------------------------------------------------\n";

        stats_file_ << "Total energy (mJ): " << (total_energy / convert_mJ) << "\n";

        // Print total time        
        total_time = (time_steps_[total_time_steps - 1].timestamp - time_steps_[0].timestamp);
        stats_file_ << "Total time: " 
                    << total_time / convert_ns_S
                    << " seconds\n";

        // Print maxiumum information
        stats_file_ << "-------------------------------------------------------------------------------------\n";

        stats_file_ << "Global maximum power reading: " << max_power 
                 << " mW on step " << max_step
                 << " at timestamp " << max_timestamp << "\n";
        
        // Print minimum information
        stats_file_ << "Global minimum power reading: " << min_power 
                 << " mW on step " << min_step
                 << " at timestamp " << min_timestamp << "\n";
        
        stats_file_ << "-------------------------------------------------------------------------------------\n";

        outfile_.close( );
        stats_file_.close(  );
    }
};

#endif /* NVMLCLASS_H_ */
