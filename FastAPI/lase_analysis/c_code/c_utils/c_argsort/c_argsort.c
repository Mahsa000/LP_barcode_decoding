#include "c_argsort.h"
#include <stdlib.h>
#include <stdint.h>


int32_t _compare_F64(const void *a, const void *b) {
  double v = ((IndexedElement_F64*)a)->value - ((IndexedElement_F64*)b)->value;
  if (v < 0) {return -1;}
  else       {return  1;}
}

int32_t _compare_F32(const void *a, const void *b) {
  float v = ((IndexedElement_F32*)a)->value - ((IndexedElement_F32*)b)->value;
  if (v < 0) {return -1;}
  else       {return  1;}
}

int32_t _compare_I32(const void *a, const void *b) {
  int32_t v = ((IndexedElement_I32*)a)->value - ((IndexedElement_I32*)b)->value;
  if (v < 0) {return -1;}
  else       {return  1;}
}


int32_t argsort_F64(double* data, int32_t* idxs, int32_t n, uint8_t reverse, uint8_t inplace) {
    int32_t ii;

    IndexedElement_F64* order_struct = (IndexedElement_F64*) malloc(n * sizeof(IndexedElement_F64));
    if(order_struct == NULL) {return -1;} // Check if malloc was successful

    for (ii=0; ii<n; ++ii) {
        order_struct[ii].index = idxs[ii];
        order_struct[ii].value = data[ii];
    }

    qsort(order_struct, n, sizeof(IndexedElement_F64), _compare_F64);

    if (reverse && inplace) {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[n-ii-1].index;
        data[ii] = order_struct[n-ii-1].value;}}
    else if (reverse) {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[n-ii-1].index;}}
    else if (inplace) {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[ii].index;
        data[ii] = order_struct[ii].value;}}
    else {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[ii].index;}}

    free(order_struct);
    return 0;
}

int32_t argsort_F32(float* data, int32_t* idxs, int32_t n, uint8_t reverse, uint8_t inplace) {
    int32_t ii;

    IndexedElement_F32* order_struct = (IndexedElement_F32*) malloc(n * sizeof(IndexedElement_F32));
    // Check if malloc was successful
    if(order_struct == NULL) {return -1;}

    for (ii=0; ii<n; ++ii) {
        order_struct[ii].index = idxs[ii];
        order_struct[ii].value = data[ii];
    }

    qsort(order_struct, n, sizeof(IndexedElement_F32), _compare_F32);

    if (reverse && inplace) {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[n-ii-1].index;
        data[ii] = order_struct[n-ii-1].value;}}
    else if (reverse) {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[n-ii-1].index;}}
    else if (inplace) {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[ii].index;
        data[ii] = order_struct[ii].value;}}
    else {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[ii].index;}}

    free(order_struct);
    return 0;
}

int32_t argsort_I32(int32_t* data, int32_t* idxs, int32_t n, uint8_t reverse, uint8_t inplace) {
    int32_t ii;

    IndexedElement_I32* order_struct = (IndexedElement_I32*) malloc(n * sizeof(IndexedElement_I32));
    // Check if malloc was successful
    if(order_struct == NULL) {return -1;}

    for (ii=0; ii<n; ++ii) {
        order_struct[ii].index = idxs[ii];
        order_struct[ii].value = data[ii];
    }

    qsort(order_struct, n, sizeof(IndexedElement_I32), _compare_I32);

    if (reverse && inplace) {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[n-ii-1].index;
        data[ii] = order_struct[n-ii-1].value;}}
    else if (reverse) {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[n-ii-1].index;}}
    else if (inplace) {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[ii].index;
        data[ii] = order_struct[ii].value;}}
    else {
      for (ii=0; ii<n; ++ii) {
        idxs[ii] = order_struct[ii].index;}}

    free(order_struct);
    return 0;
}

