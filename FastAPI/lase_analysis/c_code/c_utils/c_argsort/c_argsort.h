#ifndef C_ARGSORT_H
#define C_ARGSORT_H

#include <stdint.h>

typedef struct {
  int32_t index;
  double value;
} IndexedElement_F64;

typedef struct {
  int32_t index;
  float value;
} IndexedElement_F32;

typedef struct {
  int32_t index;
  int32_t value;
} IndexedElement_I32;


int32_t argsort_F64(double *array, int32_t *indices, int32_t length, uint8_t reverse, uint8_t inplace);

int32_t argsort_F32(float *array, int32_t *indices, int32_t length, uint8_t reverse, uint8_t inplace);

int32_t argsort_I32(int32_t *array, int32_t *indices, int32_t length, uint8_t reverse, uint8_t inplace);

#endif  // C_ARGSORT_H