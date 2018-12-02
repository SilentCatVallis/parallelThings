#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

struct edge
{
    int weight;
    int vertex;
};

struct weghtedGraph
{
    struct edge** graph;
    int* sizes;    
};

struct weghtedGraph GenerateGraph(int n, double vertexProbability)
{
    int maxWeight = 1000;
    
    struct edge** graph;
    struct edge* temp;
    int* sizes;
    
    sizes = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++)
        sizes[i] = 0;
    graph = malloc(n * sizeof(int*));
    temp = malloc(n * n * sizeof(struct edge));
    for (int i = 0; i < n; i++)
    {
        graph[i] = temp + (i * n);
        sizes[i] = 0;
    }
    
    srand ( time ( NULL));
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            double randomValue = (double)rand()/RAND_MAX;
            if (randomValue <= vertexProbability)
            {
                int weight = (int)(rand()%maxWeight);
                graph[i][sizes[i]].vertex = j;
                graph[i][sizes[i]].weight = weight;
                sizes[i] = sizes[i] + 1;
                
                graph[j][sizes[j]].vertex = i;
                graph[j][sizes[j]].weight = weight;
                sizes[j] = sizes[j] + 1;
            }
        }
    }
    
    struct weghtedGraph g;
    g.graph = graph;
    g.sizes = sizes;
    return g;
}

void FindShortestPathes(int vertexCount, int startVertex, struct weghtedGraph wGraph, int* parent, long long* pathWeight)
{
    int* visited = malloc(vertexCount * sizeof(int));
    for (int i = 0; i < vertexCount; i++)
    {
        visited[i] = 0;
        pathWeight[i] = INT_MAX;
        parent[i] = -1;
    }
    pathWeight[startVertex] = 0;
    
    int* sizes = wGraph.sizes;
    struct edge** graph = wGraph.graph;
    
    for (int i = 0; i < vertexCount; i++)
    {
        int next = -1;
        int j;
        for (j = 0; j < vertexCount; j++)
        {
            if (!(visited[j] == 1) && (next == -1 || pathWeight[j] < pathWeight[next]))
                next = j;
        }
           
        if (pathWeight[next] == INT_MAX)
            break;
        visited[next] = 1;
            
        int count = sizes[next];
        for (j = 0; j < count; j++)
        {
            int to = graph[next][j].vertex;
            int len = graph[next][j].weight;
            if (pathWeight[next] + len < pathWeight[to]) {
                pathWeight[to] = pathWeight[next] + len;
                parent[to] = next;
            }
        }
    }
    
    free(visited);
}

void FindShortestPathesParallel(int vertexCount, int startVertex, struct weghtedGraph wGraph, int* parent, long long* pathWeight)
{
    int* visited = malloc(vertexCount * sizeof(int));
    for (int i = 0; i < vertexCount; i++)
    {
        visited[i] = 0;
        pathWeight[i] = INT_MAX;
        parent[i] = -1;
    }
    pathWeight[startVertex] = 0;
    
    int* sizes = wGraph.sizes;
    struct edge** graph = wGraph.graph;
    
    //private variables
    int threadId;
    int threadsCount;
    int firstIndex;
    int lastIndex;
    
    int firstIndexForUpdate;
    int lastIndexForUpdate;
    int countForUpdate;
    
    
    //shared variables
    int minimumDistance;
    int nextVertex;
    
    # pragma omp parallel private (threadId, threadsCount, firstIndex, lastIndex, firstIndexForUpdate, lastIndexForUpdate)  shared (visited, pathWeight, graph, sizes, minimumDistance, nextVertex, countForUpdate)
    {
        threadId = omp_get_thread_num();
        threadsCount = omp_get_num_threads(); 
        firstIndex = (threadId * vertexCount) / threadsCount;
        lastIndex = ((threadId + 1) * vertexCount) / threadsCount - 1;
        
        for (int i = 0; i < vertexCount; i++)
        {
            # pragma omp single 
            {
                minimumDistance = INT_MAX;
                nextVertex = -1;
            }
            
            int localNextVertex = -1;
            int localMinimumDistance = INT_MAX;
            int j;
            for (j = firstIndex; j <= lastIndex; j++)
            {
                if (!(visited[j] == 1) && (localNextVertex == -1 || pathWeight[j] < localMinimumDistance))
                {
                    localNextVertex = j;
                    localMinimumDistance = pathWeight[j];
                }
            }
            
            # pragma omp critical
            {
                if (localMinimumDistance < minimumDistance)  
                {
                    minimumDistance = localMinimumDistance;
                    nextVertex = localNextVertex;
                }
            }
            
            # pragma omp barrier
            
            if (nextVertex == -1)
                break;
            
            # pragma omp single 
            {
                visited[nextVertex] = 1;                
                countForUpdate = sizes[nextVertex];                
            }
            
            # pragma omp barrier
            
            
            firstIndexForUpdate = (threadId * countForUpdate) / threadsCount;
            lastIndexForUpdate = ((threadId + 1) * countForUpdate) / threadsCount - 1;
                
          
            for (j = firstIndexForUpdate; j <= lastIndexForUpdate; j++)
            {
                struct edge edg = graph[nextVertex][j];
                int to = edg.vertex;
                int newLength = pathWeight[nextVertex] + edg.weight;
                if (newLength < pathWeight[to]) 
                {
                    pathWeight[to] = newLength;
                    parent[to] = nextVertex;
                }
            }
            
            #pragma omp barrier
        }
    }
    
    free(visited);
}

int main(int argc, char *argv[])
{
    double time;
    int NumThreads = omp_get_num_procs();
        
    int startVertex = 0;
    int n = 27 * 1000;
    
    if (sscanf (argv[1], "%d", &n) != 1) 
    {
        printf("error - not an integer");
    }
    
    //graph generating
    time = omp_get_wtime();
    
    struct weghtedGraph wGraph = GenerateGraph(n, 1);
    
    time = omp_get_wtime() - time;
    printf("Generating time: %f\n", time);
    
  
    struct edge** graph = wGraph.graph;
    int* sizes = wGraph.sizes; 
    
    
    //parallel code execution
    int* parentParallel = malloc(n * sizeof(int));
    long long* pathWeightParallel = malloc(n * sizeof(long long));      
       
    time = omp_get_wtime();
    
    FindShortestPathesParallel(n, startVertex, wGraph, parentParallel, pathWeightParallel);
    
    time = omp_get_wtime() - time;
    printf("%d threads calculating time: %f\n", NumThreads, time);
   
    
    
    
    
    
    //single thread code execution
    int* parentSingleThread = malloc(n * sizeof(int));
    long long* pathWeightSingleThread = malloc(n * sizeof(long long));
    
    time = omp_get_wtime();
    
    FindShortestPathes(n, startVertex, wGraph, parentSingleThread, pathWeightSingleThread);
    
    time = omp_get_wtime() - time;
    printf("Single thread calculating time: %f\n", time);
    
    
    
    
    //validation
    int correct = 1;
    for (int i = 0; i < n; i++)
    {
        if (pathWeightParallel[i] != pathWeightSingleThread[i])
        {
            correct = 0;
            printf("%lld - parallel != %lld - singleThrea (%d'th)\n", pathWeightParallel[i], pathWeightSingleThread[i], i);
        }
    }
        
    if (correct == 1)
        printf("Everything is OK");
}
