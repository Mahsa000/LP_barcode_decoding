#include "c_dict.h"
#include <stdlib.h>
#include <stdint.h>


#define MAX_DICTS 100     // Max number of dictionaries

dict_item* dictionaries[MAX_DICTS];
int NDICT = 0;


int32_t create_dict(void) {
    if (NDICT>=MAX_DICTS) {return -1;}
    dictionaries[NDICT] = NULL;
    NDICT ++;
    return NDICT-1;
}

void set_item(int32_t did, int32_t key, int32_t value) {
    dict_item *i = NULL;
    HASH_FIND_INT(dictionaries[did], &key, i);
    if (i == NULL) {
        i = malloc(sizeof(dict_item));
        i->key = key;
        HASH_ADD_INT(dictionaries[did], key, i);
    }
    i->value = value;
}

int32_t get_item(int32_t did, int32_t key, int32_t *value) {
    dict_item *i = NULL;
    HASH_FIND_INT(dictionaries[did], &key, i);
    if (i) {
        *value = i->value;
        return 0;  // key found
    }
    *value = ERR_NOTIN;
    return -1;  // key not found
}

int32_t contains_key(int32_t did, int32_t key) {
    dict_item *i = NULL;
    HASH_FIND_INT(dictionaries[did], &key, i);
    return i != NULL;
}

void delete_item(int32_t did, int32_t key) {
    dict_item* i = NULL;
    HASH_FIND_INT(dictionaries[did], &key, i);  // Retrieve the item from the hash table
    if (i != NULL) {
        HASH_DEL(dictionaries[did], i);  // Remove the item from the hash table
        free(i);                    // Free the memory for the item
    }
}

void delete_dict(int32_t did) {
  dict_item *itm, *tmp;

  HASH_ITER(hh, dictionaries[did], itm, tmp) {
    HASH_DEL(dictionaries[did], itm);  /* delete; users advances to next */
    free(itm);                         /* optional- if you want to free  */
  }
}

void delete_all(void) {
    for(int i=0; i<NDICT; i++) {
        delete_dict(i);
        dictionaries[i] = NULL;
    }
    NDICT = 0;
}