#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include <pthread.h>
#include <atomic>
#include "Barrier/Barrier.h"
#include <semaphore.h>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>

struct JobContext;

struct JobContext
{
    const MapReduceClient &client;
    const InputVec &inputVec;
    OutputVec &outputVec;
    int multiThreadLevel;
    std::vector<pthread_t> threads;
    Barrier barrier;
    std::atomic<int64_t> mapCounter;
    std::atomic<int64_t> shuffleCounter;
    std::atomic<int64_t> reduceCounter;
    JobState jobState;
    pthread_mutex_t stateMutex;
    pthread_mutex_t emitMutex;
    pthread_mutex_t waitMutex;
    std::vector<IntermediateVec> intermediateVecs;
    std::vector<IntermediateVec> shuffleQueue;
    sem_t shuffleSemaphore;

    /**
  * Constructor for JobContext
  */
    JobContext (const MapReduceClient &client, const InputVec &inputVec,
                OutputVec &outputVec, int multiThreadLevel) :
        client (client),
        inputVec (inputVec),
        outputVec (outputVec),
        multiThreadLevel (multiThreadLevel),
        barrier (multiThreadLevel),
        mapCounter (0),
        shuffleCounter(0),
        reduceCounter (0),
        jobState({UNDEFINED_STAGE,0}),
        intermediateVecs (multiThreadLevel),
        threads(multiThreadLevel)
    {
      pthread_mutex_init (&stateMutex, nullptr);
      pthread_mutex_init (&emitMutex, nullptr);
      pthread_mutex_init (&waitMutex, nullptr);
      sem_init(&shuffleSemaphore, 0, 0);
    }

    ~JobContext ()
    {
      pthread_mutex_destroy (&stateMutex);
      pthread_mutex_destroy (&emitMutex);
      sem_destroy(&shuffleSemaphore);
    }
};

struct data{
    JobContext* jc;
    bool shuffle;
    data(JobContext* jb, bool sh):
        jc(jb),
        shuffle(sh){}

};

bool compareIntermediatePairs(const IntermediatePair& a, const IntermediatePair& b) {
  return *a.first < *b.first;
}


void* mapThread(void* arg) {
  auto* jobContext = static_cast<JobContext*>(arg);
  int index;

  std::cout<<"in Map faze, beginning"<<std::endl;
  while ((index = jobContext->mapCounter.fetch_add(1)) < jobContext->inputVec.size()) {
    const InputPair& pair = jobContext->inputVec[index];
    jobContext->client.map(pair.first, pair.second, jobContext);
  }

  // Sort intermediate vectors after mapping
  auto& intermediateVec = jobContext->intermediateVecs[jobContext->mapCounter % jobContext->multiThreadLevel];
  std::sort(intermediateVec.begin(), intermediateVec.end(), compareIntermediatePairs);

  std::cout<<"in Map faze, pre barrier"<<std::endl;
  jobContext->barrier.barrier();
  return nullptr;
}

void* shuffleThread(void* arg) {
  auto* jobContext = static_cast<JobContext*>(arg);
  //jobContext->barrier.barrier();

  if (pthread_mutex_lock(&jobContext->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_lock\n";
    exit(1);
  }
  jobContext->jobState.stage = SHUFFLE_STAGE;
  if (pthread_mutex_unlock(&jobContext->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_unlock\n";
    exit(1);
  }

  // Create a map for grouped intermediate pairs
  std::unordered_map<K2*, IntermediateVec> shuffleMap;
  for (auto& vec : jobContext->intermediateVecs) {
    for (auto& pair : vec) {
      shuffleMap[pair.first].push_back(pair);
    }
  }

  for (auto& pair : shuffleMap) {
    auto& key = pair.first;
    auto& vec = pair.second;
    jobContext->shuffleQueue.push_back(vec);
    jobContext->shuffleCounter.fetch_add(1);
  }
  std::cout<<"shuffle faze: pre sep_post"<<std::endl;
  sem_post(&jobContext->shuffleSemaphore);
  return nullptr;
}

void* reduceThread(void* arg) {
  auto* jobContext = static_cast<JobContext*>(arg);
  sem_wait(&jobContext->shuffleSemaphore);

  if (pthread_mutex_lock(&jobContext->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_lock\n";
    exit(1);
  }
  jobContext->jobState.stage = REDUCE_STAGE;
  if (pthread_mutex_unlock(&jobContext->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_unlock\n";
    exit(1);
  }

  while (true) {
    int index = jobContext->reduceCounter.fetch_add(1);
    if (index >= jobContext->shuffleQueue.size()) break;

    jobContext->client.reduce(&jobContext->shuffleQueue[index], jobContext);
  }

  return nullptr;
}

void mapFunc(JobContext* jc){

  if (pthread_mutex_lock(&jc->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_lock\n";
    exit(1);
  }
  jc->jobState.stage = MAP_STAGE;
  if (pthread_mutex_unlock(&jc->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_unlock\n";
    exit(1);
  }

  int64_t index = jc->mapCounter.fetch_add(1);
  while (index < jc->inputVec.size()) {
    //std::cout<<index<<std::endl;
    const InputPair& pair = jc->inputVec[index];
    jc->client.map(pair.first, pair.second, jc);
    index = jc->mapCounter.fetch_add(1);
  }
}

void sortFunc(JobContext* jc)
{
  pthread_t currentThread = pthread_self();
  int thread_index = -1;
  // Find the index of the current thread
  for(int i = 0; i < jc->multiThreadLevel; ++i)
  {
    if(pthread_equal(jc->threads[i], currentThread))
    {
      thread_index = i;
      break;
    }
  }
  // Handle case where thread is not found
  if(thread_index == -1)
  {
    std::cout << "system error: Thread not found" << std::endl;
    exit(1);
  }
  auto &intermediateVec = jc->intermediateVecs[thread_index];

  std::sort(intermediateVec.begin(), intermediateVec.end(),
            compareIntermediatePairs);
}


void shuffleFunc(JobContext* jc){
  // so only one thread will acces the state
  if (pthread_mutex_lock(&jc->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_lock\n";
    exit(1);
  }

  jc->jobState.stage = SHUFFLE_STAGE;

  if (pthread_mutex_unlock(&jc->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_unlock\n";
    exit(1);
  }

  bool found_first = false;
  IntermediatePair max_pair;

  // find the max pair
  auto find_max_pair = [&]() -> IntermediatePair {
      found_first = false;
      IntermediatePair max_pair;
      for (const auto& vec : jc->intermediateVecs) {
        if (!vec.empty()) {
          const IntermediatePair& currentPair = vec.back();
          if (!found_first || *max_pair.first < *currentPair.first) {
            max_pair = currentPair;
            found_first = true;
          }
        }
      }
      return max_pair;
  };

  int i = 0;
  // Main loop for the shuffle phase
  while (true) {
    max_pair = find_max_pair();
    //std::cout<<i<<std::endl;
    i++;
    if (!found_first) break;
    IntermediateVec key_vec;
    for (auto& vec : jc->intermediateVecs) {
      while (!vec.empty() &&
             !(*max_pair.first < *vec.back().first) &&
             !(*vec.back().first < *max_pair.first)) {
        key_vec.push_back(vec.back());
        vec.pop_back();
        jc->shuffleCounter.fetch_add(1);
      }
      //break;
    }
    jc->shuffleQueue.push_back(key_vec);
  }
  //std::cout<<"shuffle faze: pre sep_post"<<std::endl;
  sem_post(&jc->shuffleSemaphore);
}


void reduceFunc(JobContext* jc){


  if (pthread_mutex_lock(&jc->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_lock\n";
    exit(1);
  }
  jc->jobState.stage = REDUCE_STAGE;
  if (pthread_mutex_unlock(&jc->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_unlock\n";
    exit(1);
  }

  int64_t index = jc->reduceCounter.fetch_add(1);

  while (true) {
    //std::cout<<index<<std::endl;
    if (index >= jc->shuffleQueue.size()) break;
    jc->client.reduce(&jc->shuffleQueue[index], jc);
    index = jc->reduceCounter.fetch_add(1);
  }
}

void* run_with_shuffle(void* arg){
  auto* jc = static_cast<JobContext*>(arg);
  jc->jobState.stage = MAP_STAGE;
  mapFunc(jc);
  jc->barrier.barrier();
  sortFunc(jc);
  jc->barrier.barrier();
  jc->jobState.stage = SHUFFLE_STAGE;
  shuffleFunc(jc);
  jc->barrier.barrier();
  jc->jobState.stage = REDUCE_STAGE;
  reduceFunc(jc);
  jc->barrier.barrier();
  return nullptr;
}

void* run_without_shuffle(void* arg){
  auto* jc = static_cast<JobContext*>(arg);
  mapFunc(jc);
  jc->barrier.barrier();
  sortFunc(jc);
  jc->barrier.barrier();
  reduceFunc(jc);
  jc->barrier.barrier();
  return nullptr;
}

void* one_thread(void* arg){
  auto* dat = static_cast<data*>(arg);
  auto* jc = dat->jc;
  bool shuffle = dat->shuffle;

  //jc->jobState.stage = MAP_STAGE;
  mapFunc(jc);
  jc->barrier.barrier();
  std::cout<<"Finished map"<<std::endl;
  sortFunc(jc);
  jc->barrier.barrier();
  std::cout<<"Finished sort"<<std::endl;
  if(!shuffle){
    sem_wait(&jc->shuffleSemaphore);
  }
  if(shuffle)
  {
    //jc->jobState.stage = SHUFFLE_STAGE;
    shuffleFunc(jc);
  }
  jc->barrier.barrier();
  std::cout<<"Finished shuffle"<<std::endl;
  //jc->jobState.stage = REDUCE_STAGE;
  reduceFunc(jc);
  std::cout<<"Finished reduce"<<std::endl;
  //jc->barrier.barrier();
  return nullptr;
}


JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel) {
  //Barrier br(multiThreadLevel*3);
  auto* jobContext = new JobContext(client, inputVec, outputVec,
                                    multiThreadLevel);
  for(int i = 0; i<multiThreadLevel; i++){
    auto* dat = new data(jobContext, i==0);
    if(pthread_create(&jobContext->threads[i],
                      nullptr,
                      one_thread,
                      dat) != 0)
    {
      std::cout << "system error: pthread_create\n";
      exit(1);
    }
  }
  return jobContext;
}

void emit2(K2* key, V2* value, void* context) {
  auto* jobContext = static_cast<JobContext*>(context);
//  int threadId = jobContext->mapCounter % jobContext->multiThreadLevel;
//  jobContext->intermediateVecs[threadId].emplace_back(key, value);
  pthread_t currentThread = pthread_self();
  int index = -1;
  // Find the index of the current thread
  for (int i = 0; i < jobContext->multiThreadLevel; ++i)
  {
    if (pthread_equal(jobContext->threads[i], currentThread))
    {
      index = i;
      break;
    }
  }
  // thread is not found
  if (index == -1)
  {
    std::cout << "system error: Thread not found" << std::endl;
    exit(1);
  }
  jobContext->intermediateVecs[index].emplace_back(key, value);
}

void emit3(K3* key, V3* value, void* context) {
  auto* jobContext = static_cast<JobContext*>(context);
  if (pthread_mutex_lock(&jobContext->emitMutex)!=0){
    std::cout << "system error: pthread_mutex_lock failed\n";
    exit(1);
  }
  jobContext->outputVec.emplace_back(key, value);
  if (pthread_mutex_unlock(&jobContext->emitMutex)!=0){
    std::cout << "system error: pthread_mutex_unlock failed\n";
    exit(1);
  }
}


void waitForJob(JobHandle job) {
  auto* jobContext = static_cast<JobContext*>(job);
  if (pthread_mutex_lock(&jobContext->waitMutex) != 0) {
    std::cout << "system error: pthread_mutex_lock failed\n";
    exit(1);
  }
  for (auto& thread : jobContext->threads) {
    if (pthread_join(thread, nullptr) != 0) {
      std::cout << "system error: pthread_join failed\n";
      exit(1);
    }
  }
  if (pthread_mutex_unlock(&jobContext->waitMutex) != 0) {
    std::cout << "system error: pthread_mutex_unlock failed\n";
    exit(1);
  }
}

// function to calculate progress percentage
float calculateProgressPercentage(JobContext* jobContext) {
  float percentage;
  switch (jobContext->jobState.stage) {
    case MAP_STAGE:
      percentage = (float)jobContext->mapCounter.load() / jobContext->inputVec.size();
      break;
    case SHUFFLE_STAGE:
    {
      int totalPairs = 0;
      for (const auto& vec : jobContext->intermediateVecs) {
        totalPairs += vec.size();
      }
      percentage = (float)jobContext->shuffleCounter.load() / totalPairs;
    }
      break;
    case REDUCE_STAGE:
      percentage = (float)jobContext->reduceCounter.load() / jobContext->shuffleQueue.size();
      break;
    default:
      percentage = 0.0;
  }
  return percentage * 100; // Convert to percentage
}
void getJobState(JobHandle job, JobState* state) {
  auto* jobContext = static_cast<JobContext*>(job);
  if (pthread_mutex_lock(&jobContext->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_lock failed\n";
    exit(1);
  }
  state->stage = jobContext->jobState.stage;
  state->percentage = calculateProgressPercentage(jobContext);
  if (pthread_mutex_unlock(&jobContext->stateMutex) != 0) {
    std::cout << "system error: pthread_mutex_unlock failed\n";
    exit(1);
  }
}

void closeJobHandle(JobHandle job) {
  waitForJob(job);
  auto* jobContext = static_cast<JobContext*>(job);
  delete jobContext;
}

