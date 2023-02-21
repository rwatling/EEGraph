#include "../include/graph.hpp"
#include "../include/argument_parsing.hpp"
#include "../include/gpu_error_check.cuh"
#include "../include/cuda_includes.cuh"
#include "../include/nvmlClass.cuh"
#include "../include/gpu_utils.cuh"
#include "../include/um_virtual_graph.cuh"
#include "../include/um_graph.cuh"

#include <iostream>
#include <sys/stat.h>
#include <cstdlib>
#include <unistd.h>

int main (int argc, char** argv) {
    
    const int num_benchmarks = 6;
    const int num_frameworks = 2;
    const int num_algorithms = 5;
    const int num_trials = 4;

    string benchmarks[num_benchmarks] = {"../datasets/Google/web-Google-trimmed.txt", 
                                        "../datasets/LiveJournal/soc-LiveJournal1-trimmed.txt", 
                                        "../datasets/Orkut/orkut-trimmed.el", 
                                        "../datasets/Pokec/soc-pokec-relationships.txt", 
                                        "../datasets/Road/roadNet-CA-trimmed.txt", 
                                        "../datasets/Skitter/as-skitter-trimmed.txt"}; //Dropped description headers for trimmed files
    string benchnames[num_benchmarks] = {"google", "livejournal", "orkut", "pokec", "road", "skitter"};
    string frameworks[num_frameworks] = {"classic", "um"};
    string algorithms[num_algorithms] = {"bfs", "cc", "pr", "sssp", "sswp"};

    string currentBench;
    string currentFramework;
    string currentAlg;
    string currentVariant;

    ArgumentParser arguments(argc, argv, true, false);

    for (int i = 0; i < num_benchmarks; i++) {
        currentBench = benchnames[i];

        //Read in graphs
        arguments.input = benchmarks[i];

        for (int j = 0; j < num_frameworks; j++) {
            currentFramework = frameworks[j];

            Graph graph(arguments.input, true);
            UMGraph um_graph(arguments.input,true);

            if (j == 0) {
                graph.ReadGraph();
            } else if (j == 1) {
                um_graph.ReadGraph();
            }

            for (int k = 0; k < num_algorithms * 2; k++) {
                currentAlg = algorithms[k % num_algorithms];
                
                if (k >= num_algorithms) { arguments.energy = true; }

                string trialDir;

                if (!arguments.energy) {
                    trialDir = "./" + currentAlg + "/" + currentFramework + "/" + currentBench + "/";
                } else {
                    trialDir = "./" + currentAlg + "/" + currentFramework + "-energy/" + currentBench + "/";
                }

                system(("mkdir -p " + trialDir).c_str());

                for (int l = 0; l < num_trials; l++) {

                    if (j == 0) {   //Classic
                        for (int m = 0; m < 4; m++) {
                            if (m == 0) {
                                arguments.variant = ASYNC_PUSH_TD;
                                currentVariant = "async-push-td";
                                string filename = trialDir + currentVariant + to_string(l);
                                arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
                                arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

                                system(("touch " + filename).c_str());
                                fstream file;
                                file.open(filename);

                                // Backup streambuffers of  cout
                                streambuf* stream_buffer_cout = cout.rdbuf();
                            
                                // Get the streambuffer of the file
                                streambuf* stream_buffer_file = file.rdbuf();
                            
                                // Redirect cout to file
                                cout.rdbuf(stream_buffer_file);
                            
                                cout << "Hello" << endl;
                            
                                // Redirect cout back to screen
                                cout.rdbuf(stream_buffer_cout);                                
                                file.close();
                            } else if (m == 1) {
                                arguments.variant = ASYNC_PUSH_DD;
                                currentVariant = "async-push-dd";
                                string filename = trialDir + currentVariant + to_string(l);
                                arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
                                arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);


                                system(("touch " + filename).c_str());
                                fstream file;
                                file.open(filename);

                                // Backup streambuffers of  cout
                                streambuf* stream_buffer_cout = cout.rdbuf();
                            
                                // Get the streambuffer of the file
                                streambuf* stream_buffer_file = file.rdbuf();
                            
                                // Redirect cout to file
                                cout.rdbuf(stream_buffer_file);
                            
                                cout << "Hello" << endl;
                            
                                // Redirect cout back to screen
                                cout.rdbuf(stream_buffer_cout);                                
                                file.close();
                            } else if (m == 2) {
                                arguments.variant = SYNC_PUSH_TD;
                                currentVariant = "sync-push-td";
                                string filename = trialDir + currentVariant + to_string(l);
                                arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
                                arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

                                system(("touch " + filename).c_str());
                                fstream file;
                                file.open(filename);

                                // Backup streambuffers of  cout
                                streambuf* stream_buffer_cout = cout.rdbuf();
                            
                                // Get the streambuffer of the file
                                streambuf* stream_buffer_file = file.rdbuf();
                            
                                // Redirect cout to file
                                cout.rdbuf(stream_buffer_file);
                            
                                cout << "Hello" << endl;
                            
                                // Redirect cout back to screen
                                cout.rdbuf(stream_buffer_cout);                                
                                file.close();
                            } else if (m == 3) {
                                arguments.variant = SYNC_PUSH_DD;
                                currentVariant = "sync-push-dd";
                                string filename = trialDir + currentVariant + to_string(l);
                                arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
                                arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

                                system(("touch " + filename).c_str());
                                fstream file;
                                file.open(filename);

                                // Backup streambuffers of  cout
                                streambuf* stream_buffer_cout = cout.rdbuf();
                            
                                // Get the streambuffer of the file
                                streambuf* stream_buffer_file = file.rdbuf();
                            
                                // Redirect cout to file
                                cout.rdbuf(stream_buffer_file);
                            
                                cout << "Hello" << endl;
                            
                                // Redirect cout back to screen
                                cout.rdbuf(stream_buffer_cout);                                
                                file.close();
                            }
                        }
                    } else if (j == 1) {    //UM
                        for (int m = 0; m < 4; m++) {
                            if (m == 0) {
                                arguments.variant = ASYNC_PUSH_TD;
                                currentVariant = "um-async-push-td";
                                string filename = trialDir + currentVariant + to_string(l);
                                arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
                                arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

                                system(("touch " + filename).c_str());
                                fstream file;
                                file.open(filename);

                                // Backup streambuffers of  cout
                                streambuf* stream_buffer_cout = cout.rdbuf();
                            
                                // Get the streambuffer of the file
                                streambuf* stream_buffer_file = file.rdbuf();
                            
                                // Redirect cout to file
                                cout.rdbuf(stream_buffer_file);
                            
                                cout << "Hello" << endl;
                            
                                // Redirect cout back to screen
                                cout.rdbuf(stream_buffer_cout);                                
                                file.close();
                            } else if (m == 1) {
                                arguments.variant = ASYNC_PUSH_DD;
                                currentVariant = "um-async-push-dd";
                                string filename = trialDir + currentVariant + to_string(l);
                                arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
                                arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

                                system(("touch " + filename).c_str());
                                fstream file;
                                file.open(filename);

                                // Backup streambuffers of  cout
                                streambuf* stream_buffer_cout = cout.rdbuf();
                            
                                // Get the streambuffer of the file
                                streambuf* stream_buffer_file = file.rdbuf();
                            
                                // Redirect cout to file
                                cout.rdbuf(stream_buffer_file);
                            
                                cout << "Hello" << endl;
                            
                                // Redirect cout back to screen
                                cout.rdbuf(stream_buffer_cout);                                
                                file.close();
                            } else if (m == 2) {
                                arguments.variant = SYNC_PUSH_TD;
                                currentVariant = "um-sync-push-td";
                                string filename = trialDir + currentVariant + to_string(l);
                                arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
                                arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

                                system(("touch " + filename).c_str());
                                fstream file;
                                file.open(filename);

                                // Backup streambuffers of  cout
                                streambuf* stream_buffer_cout = cout.rdbuf();
                            
                                // Get the streambuffer of the file
                                streambuf* stream_buffer_file = file.rdbuf();
                            
                                // Redirect cout to file
                                cout.rdbuf(stream_buffer_file);
                            
                                cout << "Hello" << endl;
                            
                                // Redirect cout back to screen
                                cout.rdbuf(stream_buffer_cout);                                
                                file.close();
                            } else if (m == 3) {
                                arguments.variant = SYNC_PUSH_DD;
                                currentVariant = "um-sync-push-dd";
                                string filename = trialDir + currentVariant + to_string(l);
                                arguments.energyFile = trialDir + currentVariant + "-readings" + to_string(l);
                                arguments.energyStats = trialDir  + currentVariant + "-readings" + to_string(l);

                                system(("touch " + filename).c_str());
                                fstream file;
                                file.open(filename);

                                // Backup streambuffers of  cout
                                streambuf* stream_buffer_cout = cout.rdbuf();
                            
                                // Get the streambuffer of the file
                                streambuf* stream_buffer_file = file.rdbuf();
                            
                                // Redirect cout to file
                                cout.rdbuf(stream_buffer_file);
                            
                                cout << "Hello" << endl;
                            
                                // Redirect cout back to screen
                                cout.rdbuf(stream_buffer_cout);                                
                                file.close();
                            }
                        }
                    }
                }
            }

            if (j == 1) {
                gpuErrorcheck(cudaFree(um_graph.edges));
                gpuErrorcheck(cudaFree(um_graph.weights));
            }
        }
    }
}