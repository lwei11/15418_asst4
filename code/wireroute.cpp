#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>

#include <mpi.h>

#include "wireroute.h"
#define ROOT 0

void print_stats(const std::vector<std::vector<int>>& occupancy) {
  int max_occupancy = 0;
  long long total_cost = 0;

  for (const auto& row : occupancy) {
    for (const int count : row) {
      max_occupancy = std::max(max_occupancy, count);
      total_cost += count * count;
    }
  }

  std::cout << "Max occupancy: " << max_occupancy << '\n';
  std::cout << "Total cost: " << total_cost << '\n';
}

void write_output(const std::vector<Wire>& wires, const int num_wires, const std::vector<std::vector<int>>& occupancy, const int dim_x, const int dim_y, const int nproc, std::string input_filename) {
  if (std::size(input_filename) >= 4 && input_filename.substr(std::size(input_filename) - 4) == ".txt") {
    input_filename.resize(std::size(input_filename) - 4);
  }

  const std::string occupancy_filename = input_filename + "_occupancy_" + std::to_string(nproc) + ".txt";
  const std::string wires_filename = input_filename + "_wires_" + std::to_string(nproc) + ".txt";

  std::ofstream out_occupancy(occupancy_filename, std::fstream::out);
  if (!out_occupancy) {
    std::cerr << "Unable to open file: " << occupancy_filename << '\n';
    exit(EXIT_FAILURE);
  }

  out_occupancy << dim_x << ' ' << dim_y << '\n';
  for (const auto& row : occupancy) {
    for (const int count : row) {
      out_occupancy << count << ' ';
    }
    out_occupancy << '\n';
  }

  out_occupancy.close();

  std::ofstream out_wires(wires_filename, std::fstream:: out);
  if (!out_wires) {
    std::cerr << "Unable to open file: " << wires_filename << '\n';
    exit(EXIT_FAILURE);
  }

  out_wires << dim_x << ' ' << dim_y << '\n' << num_wires << '\n';

  for (const auto& [start_x, start_y, end_x, end_y, bend1_x, bend1_y] : wires) {
    out_wires << start_x << ' ' << start_y << ' ' << bend1_x << ' ' << bend1_y << ' ';

    if (start_y == bend1_y) {
    // first bend was horizontal

      if (end_x != bend1_x) {
        // two bends

        out_wires << bend1_x << ' ' << end_y << ' ';
      }
    } else if (start_x == bend1_x) {
      // first bend was vertical

      if (end_y != bend1_y) {
        // two bends

        out_wires << end_x << ' ' << bend1_y << ' ';
      }
    }
    out_wires << end_x << ' ' << end_y << '\n';
  }

  out_wires.close();
}

void init_random(Wire& currWire) {
  int x0 = currWire.start_x;
  int x1 = currWire.end_x;
  int y0 = currWire.start_y;
  int y1 = currWire.end_y;
  srand(std::chrono::steady_clock::now().time_since_epoch().count());
  if (x0 == x1 || y0 == y1) 
    return;
  int xdist = abs(x0 - x1);
  int ydist = abs(y0 - y1);
  if (rand() % 2 == 1) {
    int bend_loc = rand() % xdist + 1 + ((x0 > x1) ? x1 : x0);
    currWire.bend1_x = (x0 > x1)? bend_loc -1 : bend_loc;
    currWire.bend1_y = y0;
    // printf("Entered X: X0 is %d, X1 is %d, Y0 is %d, Y1 is %d, bend1 is (%d, %d)\n", x0, x1, y0, y1, currWire.bend1_x, currWire.bend1_y);
  }
  else {
    int bend_loc = rand() % ydist + 1 + ((y0 > y1) ? y1 : y0);
    currWire.bend1_y = (y0 > y1)? bend_loc -1 : bend_loc;
    currWire.bend1_x = x0;
    // printf("ENTERED Y: X0 is %d, X1 is %d, Y0 is %d, Y1 is %d, bend1 is (%d, %d)\n", x0, x1, y0, y1, currWire.bend1_x, currWire.bend1_y);
  }
}

void update_occupancy(Wire& wire, std::vector<std::vector<int>>& occupancy, int increment) {
  int x0 = wire.start_x;
  int x1 = wire.end_x;
  int y0 = wire.start_y;
  int y1 = wire.end_y;
  int sX = (x0 > x1) ? x1 : x0;
  int eX = (x0 > x1) ? x0 : x1;
  bool xflipped = (x0 > x1);
  int sY = (y0 > y1) ? y1 : y0;
  int eY = (y0 > y1) ? y0 : y1;
  bool yflipped = (y0 > y1);
  if( wire.bend1_x == wire.start_x) {
    for(int i=sX; i<=eX; i++) { 
      occupancy[wire.bend1_y][i] += increment;
    }
    for(int j=sY; j<=eY; j++) {
      if (j<wire.bend1_y && (xflipped != yflipped)) {
        occupancy[j][eX] += increment;
      }
      else if (j<wire.bend1_y && (xflipped == yflipped)) {
        occupancy[j][sX] += increment;
      }
      if (j>wire.bend1_y && (xflipped != yflipped)) {
        occupancy[j][sX] += increment;
      } 
      else if (j>wire.bend1_y && (xflipped == yflipped)) {
        occupancy[j][eX] += increment;
      }
    }
  }
  else {
    for(int i=sX; i<=eX; i++) {
      if (i<wire.bend1_x && (xflipped != yflipped)) {
        occupancy[eY][i] += increment;
      } 
      else if (i<wire.bend1_x && (xflipped == yflipped)) {
        occupancy[sY][i] += increment;
      }
      if (i>wire.bend1_x && (xflipped != yflipped)) {
        occupancy[sY][i] += increment;
      } 
      else if (i>wire.bend1_x && (xflipped == yflipped)) {
        occupancy[eY][i] += increment;
      }
    }
    for(int j=sY; j<=eY; j++) {
      occupancy[j][wire.bend1_x] += increment;
    }
  }
}

void compute_occupancy(std::vector<Wire>& wires, std::vector<std::vector<int>>& occupancy) {
  for(auto& wire: wires) {
    // printf("Wire # %d\n", counter);
    // printf("Bend is at (%d, %d)\n", wire.bend1_x, wire.bend1_y);
    int x0 = wire.start_x;
    int x1 = wire.end_x;
    int y0 = wire.start_y;
    int y1 = wire.end_y;
    int sX = (x0 > x1) ? x1 : x0;
    int eX = (x0 > x1) ? x0 : x1;
    bool xflipped = (x0 > x1);
    int sY = (y0 > y1) ? y1 : y0;
    int eY = (y0 > y1) ? y0 : y1;
    bool yflipped = (y0 > y1);
    if( wire.bend1_x == wire.start_x) {  // bend vertically (first go along y axis)
      for(int i=sX; i<=eX; i++) { 
        // update all x-axis (1 horizontal line segment)
        // printf("1 : Updated (%d, %d)\n", i, wire.bend1_y);
        occupancy[wire.bend1_y][i] += 1;
      }
      for(int j=sY; j<=eY; j++) {
        // update all y-axis (2 vertical line segments)
        if (j<wire.bend1_y && (xflipped != yflipped)) {
          // printf("2 : Updated (%d, %d)\n", eX, j);
          occupancy[j][eX] += 1;  // line segment before the bend
        }
        else if (j<wire.bend1_y && (xflipped == yflipped)) {
          // printf("2 : Updated (%d, %d)\n", sX, j);
          occupancy[j][sX] += 1;  // line segment before the bend
        }
        if (j>wire.bend1_y && (xflipped != yflipped)) {
          // printf("3 : Updated (%d, %d)\n", sX, j);
          occupancy[j][sX] += 1;  // line segment after the bend
        } 
        else if (j>wire.bend1_y && (xflipped == yflipped)) {
          // printf("3 : Updated (%d, %d)\n", eX, j);
          occupancy[j][eX] += 1;  // line segment after the bend
        }
      }
    } // bend vertically (first go along y axis)
    else {
      for(int i=sX; i<=eX; i++) { // bend horizontally (first go along x axis)
        // update all x-axis (2 horizontal line segments)
        if (i<wire.bend1_x && (xflipped != yflipped)) {
          // printf("4 : Updated (%d, %d)\n", i, sY);
          occupancy[eY][i] += 1; // line segment before the bend
        } 
        else if (i<wire.bend1_x && (xflipped == yflipped)) {
          // printf("4 : Updated (%d, %d)\n", i, sY);
          occupancy[sY][i] += 1; // line segment before the bend
        }
        if (i>wire.bend1_x && (xflipped != yflipped)) {
          // printf("5 : Updated (%d, %d)\n", i, eY);
          occupancy[sY][i] += 1;  // line segment after the bend
        } 
        else if (i>wire.bend1_x && (xflipped == yflipped)) {
          // printf("5 : Updated (%d, %d)\n", i, eY);
          occupancy[eY][i] += 1;  // line segment after the bend
        }
      }
      for(int j=sY; j<=eY; j++) {
        // update all y-axis (1 vertical line segment)
        // printf("6 : Updated (%d, %d)\n", wire.bend1_x, j);
        occupancy[j][wire.bend1_x] += 1;
      }
    } // bend horizontally (first go along x axis)
  }
}

int compute_potential_cost(int x0, int x1, int y0, int y1, int bx, int by, std::vector<std::vector<int>>& occupancy) {
  int cost = 0;
  int sX = (x0 > x1) ? x1 : x0;
  int eX = (x0 > x1) ? x0 : x1;
  bool xflipped = (x0 > x1);
  int sY = (y0 > y1) ? y1 : y0;
  int eY = (y0 > y1) ? y0 : y1;
  bool yflipped = (y0 > y1);
  if(bx == x0) {
    for(int i=sX; i<=eX; i++) { 
      cost += occupancy[by][i];
    }
    for(int j=sY; j<=eY; j++) {
      if (j<by && (xflipped != yflipped)) {
        cost += occupancy[j][eX];
      }
      else if (j<by && (xflipped == yflipped)) {
        cost += occupancy[j][sX];
      }
      if (j>by && (xflipped != yflipped)) {
        cost += occupancy[j][sX];
      } 
      else if (j>by && (xflipped == yflipped)) {
        cost += occupancy[j][eX];
      }
    }
  }
  else {
    for(int i=sX; i<=eX; i++) {
      if (i<bx && (xflipped != yflipped)) {
        cost += occupancy[eY][i];
      } 
      else if (i<bx && (xflipped == yflipped)) {
        cost += occupancy[sY][i];
      }
      if (i>bx && (xflipped != yflipped)) {
        cost += occupancy[sY][i];
      } 
      else if (i>bx && (xflipped == yflipped)) {
        cost += occupancy[eY][i];
      }
    }
    for(int j=sY; j<=eY; j++) {
      cost += occupancy[j][bx];
    }
  }
  return cost;
}

Wire compute_shortest_path_across(Wire& wire, std::vector<std::vector<int>>& occupancy, long long currCost, const double SA_prob) {
  int x0 = wire.start_x;
  int x1 = wire.end_x;
  int y0 = wire.start_y;
  int y1 = wire.end_y;
  int rX = wire.bend1_x;
  int rY = wire.bend1_y;
  int sX = (x0 > x1) ? x1 : x0;
  int eX = (x0 > x1) ? x0 : x1;
  bool xflipped = (x0 > x1);
  int sY = (y0 > y1) ? y1 : y0;
  int eY = (y0 > y1) ? y0 : y1;
  bool yflipped = (y0 > y1);
  long long finCost = currCost;

  {
    update_occupancy(wire, occupancy, -1);
  }

  if (rand() % 100 < (SA_prob * 100)) {
    init_random(wire);
    rX = wire.bend1_x;
    rY = wire.bend1_y;
  }
  else {
    long long tempCost;
    
    //Shortest horizontal path
    for (int i=1;i<=eX-sX;i++) {
      int bendx = xflipped ? (eX - i) : (sX + i);
      tempCost = compute_potential_cost(x0, x1, y0, y1, bendx, y0, occupancy);
      if (tempCost < finCost) {
        finCost = tempCost;
        rX = bendx;
        rY = y0;
      }
    }

    //Shortest vertical path
    for (int j=1;j<=eY-sY;j++) {
      int bendy = yflipped ? (eY - j) : (sY + j);
      tempCost = compute_potential_cost(x0, x1, y0, y1, x0, bendy, occupancy);
      if (tempCost < finCost) {
        finCost = tempCost;
        rX = x0;
        rY = bendy;
      }
    }
    
    //Update Wire and Matrix
  } 

  wire.bend1_x = rX;
  wire.bend1_y = rY;
  {
    update_occupancy(wire, occupancy, 1);
  }
  return wire;
}

void reset_occupancy(const int dim_x, const int dim_y, std::vector<std::vector<int>>& occupancy) {
  for(int i=0; i<dim_x; i++) {
    for(int j=0; j<dim_y; j++) {
      occupancy[i][j] = 0;
    }
  }
}

int main(int argc, char *argv[]) {
  const auto init_start = std::chrono::steady_clock::now();
  int pid;
  int nproc;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  // Get total number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Status status;

  // f("Curr PID: %d\n", pid);

  std::string input_filename;
  double SA_prob = 0.1;
  int SA_iters = 5;
  char parallel_mode = '\0';
  int batch_size = 1;

  // Read command line arguments
  int opt;
  while ((opt = getopt(argc, argv, "f:p:i:m:b:")) != -1) {
    switch (opt) {
      case 'f':
        input_filename = optarg;
        break;
      case 'p':
        SA_prob = atof(optarg);
        break;
      case 'i':
        SA_iters = atoi(optarg);
        break;
      case 'm':
        parallel_mode = *optarg;
        break;
      case 'b':
        batch_size = atoi(optarg);
        break;
      default:
        if (pid == ROOT) {
          std::cerr << "Usage: " << argv[0] << " -f input_filename [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
  }

  // Check if required options are provided
  if (empty(input_filename) || SA_iters <= 0 || (parallel_mode != 'A' && parallel_mode != 'W') || batch_size <= 0) {
    if (pid == ROOT) {
      std::cerr << "Usage: " << argv[0] << " -f input_filename [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (pid == ROOT) {
    std::cout << "Number of processes: " << nproc << '\n';
    std::cout << "Simulated annealing probability parameter: " << SA_prob << '\n';
    std::cout << "Simulated annealing iterations: " << SA_iters << '\n';
    std::cout << "Input file: " << input_filename << '\n';
    std::cout << "Parallel mode: " << parallel_mode << '\n';
    std::cout << "Batch size: " << batch_size << '\n';
  }

  int dim_x, dim_y, num_wires;
  std::vector<Wire> wires;
  std::vector<std::vector<int>> occupancy;

  if (pid == ROOT) {
      std::ifstream fin(input_filename);

      if (!fin) {
        std::cerr << "Unable to open file: " << input_filename << ".\n";
        exit(EXIT_FAILURE);
      }

      /* Read the grid dimension and wire information from file */
      fin >> dim_x >> dim_y >> num_wires;

      wires.resize(num_wires);
      for (auto& wire : wires) {
        fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
        wire.bend1_x = wire.start_x;
        wire.bend1_y = wire.start_y;
      }
  }

  /* Initialize any additional data structures needed in the algorithm */

  if (pid == ROOT) {
    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';
  }

  const auto compute_start = std::chrono::steady_clock::now();

  /** 
   * (TODO)
   * Implement the wire routing algorthm here
   * Feel free to structure the algorithm into different functions
   * Use MPI to parallelize the algorithm. 
   */
  MPI_Bcast(&dim_x, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&dim_y, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&num_wires, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

  occupancy.resize(dim_y, std::vector<int>(dim_x));
  wires.resize(num_wires);

  if (pid == ROOT) {
    for(auto& wire: wires) {
      init_random(wire);
    }
  }

  MPI_Bcast(wires.data(), 6 * num_wires, MPI_INT, ROOT, MPI_COMM_WORLD);

  compute_occupancy(wires, occupancy);

  int *counts = (int*)calloc(nproc, sizeof(int));
  int rem = (num_wires)%nproc;

  for (int k = 0; k < nproc; k++) {
      counts[k] = num_wires/nproc;
      counts[k] *= 6;
      if (rem > 0) {
          counts[k]+=6;
          rem--;
      }
  }
  
  std::vector<Wire> node_wires(counts[pid] / 6);

  std::vector<Wire> new_batch_wires(batch_size);
  std::vector<Wire> old_batch_wires(batch_size);

  int c = 0;
  for (int test = pid; test<num_wires; test+=nproc) {
    node_wires[c] = wires[test];
    c += 1;
  }
  
  int num_batches = num_wires / (nproc * batch_size);
  //printf("Num batches is %d\n", num_batches);
  if (num_wires % (nproc * batch_size) != 0) {
    num_batches += 1;
  }

  for (int i = 0; i < SA_iters; i++) {
    int curr_batch = 0;
    while (curr_batch < num_batches) {
      int start_wire = curr_batch * batch_size;
      int num_wires_to_process = batch_size;
      if (start_wire + num_wires_to_process > counts[pid]/6) {
        num_wires_to_process = counts[pid]/6 - start_wire;
        //printf("Remainder is %d\n", num_wires_to_process);
      }
      Wire batch_wire;
      Wire new_batch_wire;
      for (int j = 0;j<num_wires_to_process;j++) {
        batch_wire = node_wires[j + start_wire];
        int curr_batch_cost = compute_potential_cost(batch_wire.start_x, batch_wire.end_x, batch_wire.start_y, batch_wire.end_y, batch_wire.bend1_x, batch_wire.bend1_y, occupancy);
        new_batch_wire = compute_shortest_path_across(batch_wire, occupancy, curr_batch_cost, SA_prob);
        old_batch_wires[j] = new_batch_wire;
        node_wires[j + start_wire] = new_batch_wire;
      }

      int curr_src = pid;
      // Wire new_wire;
      // for (auto& wire: node_wires) {
      //   int currCost = compute_potential_cost(wire.start_x, wire.end_x, wire.start_y, wire.end_y, wire.bend1_x, wire.bend1_y, occupancy);
      //   new_wire = compute_shortest_path_across(wire, occupancy, currCost, SA_prob);
      //   old_wires[counter] = new_wire;
      //   counter += 1;
      // }

      for (int idx = 0; idx<num_wires_to_process; idx+=1) {
        int offset = curr_batch * batch_size * nproc + curr_src;
        wires[idx * nproc + offset] = old_batch_wires[idx];
      }
      
      
      for (int i=0;i<nproc-1;i++) {
        MPI_Request reqsend, reqrecv;
        int dst = (pid + 1) % nproc;
        int src = (pid - 1 + nproc) % nproc;
        curr_src -= 1;
        if (curr_src < 0) {
          curr_src = nproc - 1;
        }
        if (start_wire + num_wires_to_process > counts[curr_src]/6) {
          num_wires_to_process = counts[curr_src]/6 - start_wire;
          //printf("Remainder is %d\n", num_wires_to_process);
        }

        int curr_batch_offset = curr_batch * batch_size * nproc + curr_src;
        
        // if (pid == ROOT) {
        //   printf("Curr src is %d\n", curr_src);
        //   printf("Curr offset is %d\n", curr_batch_offset);
        // }

        //const auto msg_start = std::chrono::steady_clock::now();
        if(pid % 2 == 0)
        {
            MPI_Isend(old_batch_wires.data(), batch_size * 6, MPI_INT, dst, 0, MPI_COMM_WORLD, &reqsend);
            MPI_Irecv(new_batch_wires.data(), batch_size * 6, MPI_INT, src, 0, MPI_COMM_WORLD, &reqrecv);
        }
        else
        {
            MPI_Irecv(new_batch_wires.data(), batch_size * 6, MPI_INT, src, 0, MPI_COMM_WORLD, &reqrecv);
            MPI_Isend(old_batch_wires.data(), batch_size * 6, MPI_INT, dst, 0, MPI_COMM_WORLD, &reqsend);
        }
        //Update using old wires
        //Wait for asynchronous commands to complete
        // for (int idx = curr_src; idx<num_wires; idx+=nproc) {
        //   update_occupancy(wires[idx], occupancy, -1);
        // }

        for (int idx = 0; idx<num_wires_to_process; idx+=1) {
          // if (pid == ROOT) {
          //   printf("Curr wire update is at %d\n", idx * nproc + curr_batch_offset);
          //   printf("Num wires is %d\n", num_wires);
          // }
          update_occupancy(wires[idx * nproc + curr_batch_offset], occupancy, -1);
        }

        MPI_Wait(&reqrecv, &status);

        //Set old wires = new wires

        // int new_counter = 0;
        // for (int idx = curr_src; idx<num_wires; idx+=nproc) {
        //   wires[idx] = new_wires[new_counter];
        //   old_wires[new_counter] = new_wires[new_counter];
        //   update_occupancy(wires[idx], occupancy, 1);
        //   new_counter += 1;
        // }

        for (int idx = 0; idx<num_wires_to_process; idx+=1) {
          wires[idx * nproc + curr_batch_offset] = new_batch_wires[idx];
          old_batch_wires[idx] = new_batch_wires[idx];
          update_occupancy(wires[idx * nproc + curr_batch_offset], occupancy, 1);
        }

        //const double msg_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - msg_start).count();
        // if (pid < 1) {
        //   std::cout << "Message " << i << " has time (sec): " << std::fixed << std::setprecision(10) << msg_time << '\n';
        // }
      }

      
      // if (pid == ROOT) {
      //   printf("Curr batch is %d\n", curr_batch);
      //   printf("Curr start is %d\n", start_wire);
      //   printf("Num wires is %d\n", num_wires_to_process);
      // }
      start_wire += batch_size;
      curr_batch += 1;

      // if (pid == ROOT) {
      //   printf("--------\n");
      // }
    }
  }

  if (pid == ROOT) {
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << std::fixed << std::setprecision(10) << compute_time << '\n';

    /* Write wires and occupancy matrix to files */
    print_stats(occupancy);
    write_output(wires, num_wires, occupancy, dim_x, dim_y, nproc, input_filename);
  }

  free(counts);
  // Cleanup
  MPI_Finalize();
}