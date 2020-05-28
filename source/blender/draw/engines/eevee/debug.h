#ifndef _DEBUG_H
#define	_DEBUG_H

#ifdef DEBUG
#define DEBUG_TEST 1
#else
#define DEBUG_TEST 0
#endif

#define dbg_printf(...) \
  do { if (DEBUG_TEST) printf(__VA_ARGS__); } while (0)

#endif
