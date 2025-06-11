#ifndef C_DICT_H
#define C_DICT_H

#include <stdint.h>
#include "uthash.h"

#define ERR_NOTIN INT32_MIN

typedef struct {
    int32_t key;
    int32_t value;
    UT_hash_handle hh;
} dict_item;

int32_t create_dict(void);
void set_item(int32_t did, int32_t key, int32_t value);
int32_t get_item(int32_t did, int32_t key, int32_t *value);
void delete_item(int32_t did, int32_t key);
int32_t contains_key(int32_t did, int32_t key);
void delete_dict(int32_t did);
void delete_all(void);

#endif  // C_DICT_H