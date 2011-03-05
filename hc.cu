#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <cutil_inline.h>
#include <cudpp/cudpp.h>

// Liczba punktów
const int n = 5;

// Liczba wymiarów
const int d = 2;

__global__ void calculate_distances(float* d_vectors, float* d_distance) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;

  int index = i + j*n;
  d_distance[index] = 0;

  // sprawdzenie czy nie przekraczamy "n" oraz rozpatrujemy tylko dolną część tablicy dwuwymiarowej
  if (i<n && j<n && i<j) {  
    for (int k=0; k<d; k++) {
      // obliczenie odległości euklidesowej (bez pierwiastka - nie potrzebny do obliczeń)
      float r = d_vectors[i*d + k] - d_vectors[j*d + k];
      d_distance[index] += r*r;
    }
  }
}

__global__ void merge_clusters(float* d_distance, int* d_dendrogram, int* d_merged_clusters, int step) {
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  
  if (x == 0 && y == 0) {
    // znajdź najmniejszą wartość w macierzy odległości
    int min_index = INT_MAX;
    for (int i=0; i<n-1; i++) {
      for (int j=i+1; j<n; j++) {
        // wez pod uwagę jeżeli klaster "i" i klaster "j" nie są złączone
        if (!d_merged_clusters[i] && !d_merged_clusters[j]) {
          int index = i + j*n;
          if (min_index == INT_MAX || d_distance[index] < d_distance[min_index])
            min_index = index;
        }
      }
    }
    
    if (min_index != INT_MAX) {
      int i = min_index/n;
      int j = min_index%n;
  
      // zapisz połączone klastry
      d_dendrogram[step] = i;
      d_dendrogram[step+(n-1)] = j;
      
      // wyłącz jeden z klastrów z dalszym obliczeń
      d_merged_clusters[j] = 1;
    }
  }
}

__global__ void update_distances(float* d_distance, int* d_dendrogram, int* d_merged_clusters, int step) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;

  int index = i + j*n;

  // sprawdzenie czy nie przekraczamy "n" oraz rozpatrujemy tylko dolną część tablicy dwuwymiarowej
  if (i<n && j<n && i<j) {
    // aktualizujemy tablice tylko dla klastrów, które nie są scalone
    if (!d_merged_clusters[i] && !d_merged_clusters[j]) {
      int candidate_index = INT_MAX;

      if (d_dendrogram[step] == i)
        candidate_index = d_dendrogram[step+(n-1)] + j*n;
      else if (d_dendrogram[step] == j)
        candidate_index = d_dendrogram[step+(n-1)] + i*n;
      
      if (candidate_index != INT_MAX && d_distance[candidate_index] < d_distance[index])
        d_distance[index] = d_distance[candidate_index];
      
    }
  }
}

void print_step_results(int step, float* h_distance, int* h_dendrogram, int* h_merged_clusters) {
  printf("\n\n\n");    
  printf("Krok %i", step+1);
  printf("\n\n");
  
  printf("Macierz odległości:");
  for (int i=0; i<n; i++) {
    printf("\n");
    for (int j=0; j<n; j++) {
      if (!h_merged_clusters[i] && !h_merged_clusters[j])
        printf(" %f ",h_distance[i*n+j]);
      else
        printf("     M     ");
    }
  }
  printf("\n\n");

  printf("Klastry złączone:");
  printf("\n");
  for (int i=0;i<n;i++) {
    printf(" C%i ",i);
  }
  printf("\n");
  for (int i=0;i<n;i++) {
    printf(" %i ",h_merged_clusters[i]);
  }
  printf("\n\n");

  printf("Dendrogram:");
  for (int i=0;i<(n-1)*2;i++) {
    if (i%(n-1)==0)
      printf("\n");
    printf(" %i ",h_dendrogram[i]);
  }
  printf("\n\n");
}


int main(int argc, char** argv) {
  if (cutCheckCmdLineFlag(argc, (const char**)argv, "device"))
    cutilDeviceInit(argc, argv);
  else
    cudaSetDevice(cutGetMaxGflopsDeviceId());

  float* h_vectors=(float*)malloc(sizeof(float)* n*d);
  float* h_distance=(float*)malloc(sizeof(float)* n*n);
  int* h_dendrogram=(int*)malloc(sizeof(int)* (n-1)*2);
  int* h_merged_clusters=(int*)malloc(sizeof(int)* n);  

  float* d_vectors;
  cutilSafeCall(cudaMalloc((void**)&d_vectors,sizeof(float)* n*d));
  
  float* d_distance;
  cutilSafeCall(cudaMalloc((void**)&d_distance,sizeof(float)* n*n));
  
  int* d_dendrogram;
  cutilSafeCall(cudaMalloc((void**)&d_dendrogram,sizeof(int)* (n-1)*2));
  
  int* d_merged_clusters;
  cutilSafeCall(cudaMalloc((void**)&d_merged_clusters,sizeof(int)* n));

  // wypełnij wektory losowymi wartościami
  // for (int i=0;i<n*d;i++) {
  //   h_vectors[i] = rand()%10;
  // }
  
  // P0
  h_vectors[0*d] = -5;
  h_vectors[0*d+1] = 4;
  
  // P1
  h_vectors[1*d] = 4;
  h_vectors[1*d+1] = -3;
  
  // P2
  h_vectors[2*d] = 5;
  h_vectors[2*d+1] = -5;
  
  // P3
  h_vectors[3*d] = -3;
  h_vectors[3*d+1] = 5;
  
  // P4
  h_vectors[4*d] = 1;
  h_vectors[4*d+1] = 1;
  
  // wypełnij tablice dendrogramu zerami
  for (int i=0;i<(n-1)*2;i++) {
    h_dendrogram[i] = 0;
  }
  
  for (int i=0;i<n;i++) {
    h_merged_clusters[i] = 0;
  }
  
  printf("Punkty wejściowe:");
  for (int i=0;i<n*d;i++) {
    if (i%d==0) {
      printf("\n");
      printf("P%i:", i/d);
    }
    printf(" %f ",h_vectors[i]);
  }
  printf("\n\n\n");
  
  // skopiuj tablice punktów z hosta do karty graficznej
  cutilSafeCall(cudaMemcpy(d_vectors,h_vectors,n*d*sizeof(float),cudaMemcpyHostToDevice));
  
  // skopiuj tablice dendrogramu z hosta do karty graficznej
  cutilSafeCall(cudaMemcpy(d_dendrogram,h_dendrogram,(n-1)*2*sizeof(int),cudaMemcpyHostToDevice));
  
  // skopiuj tablice scalonych klastrów z hosta do karty graficznej
  cutilSafeCall(cudaMemcpy(d_merged_clusters,h_merged_clusters,n*sizeof(int),cudaMemcpyHostToDevice));

  // calculate grid width and height
  int width=n/16+(((n%16)!=0)?1:0);
  int height=n/16+(((n%16)!=0)?1:0);

  dim3 grid(width,height);
  dim3 block(16,16);
  
  dim3 sgrid(1,1);
  dim3 sblock(4,4);

  // oblicz macierz odległości
  calculate_distances<<<grid,block>>>(d_vectors, d_distance);
    
  // skopiuj macierz odległości do hosta
  cutilSafeCall(cudaMemcpy(h_distance,d_distance,n*n*sizeof(float), cudaMemcpyDeviceToHost));
  
  printf("Macierz odległości:");
  for (int i=0;i<n*n;i++) {
    if (i%n==0)
      printf("\n");
    printf(" %f ",h_distance[i]);
  }
  printf("\n\n");
  
  cutilSafeCall(cudaMemcpy(d_dendrogram,h_dendrogram,(n-1)*2*sizeof(int),cudaMemcpyHostToDevice));
  
  for (int step=0; step < n-1; step++){
    
    // scal klastry dla najmniejszej odległości
    merge_clusters<<<sgrid,sblock>>>(d_distance, d_dendrogram, d_merged_clusters, step);
    
    // zaktualizuj macierz odległości z uwzględnieniem scalonych klastrów
    update_distances<<<grid,block>>>(d_distance, d_dendrogram, d_merged_clusters, step);
    
    // prześlij obliczenia na host (tylko w celu wyświetlenia obliczeń dla kroku)
    cutilSafeCall(cudaMemcpy(h_distance,d_distance,n*n*sizeof(float), cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(h_dendrogram,d_dendrogram,(n-1)*2*sizeof(int), cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(h_merged_clusters,d_merged_clusters,n*sizeof(int), cudaMemcpyDeviceToHost));
    
    // wyświetl wyniki dla danego kroku
    print_step_results(step, h_distance, h_dendrogram, h_merged_clusters);
    
  }
  
  cudaThreadExit();
}