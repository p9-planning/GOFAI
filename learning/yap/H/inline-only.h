#ifndef _YAP_INLINE_ONLY_H_
#define _YAP_INLINE_ONLY_H_

#ifdef __GNUC__
#define INLINE_ONLY extern inline __attribute__((gnu_inline,always_inline))
//#define INLINE_ONLY
#else
#define INLINE_ONLY  EXTERN
#endif

#endif
