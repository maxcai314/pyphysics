#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

int __sprintf_chk(char* s, int flag, size_t slen, const char* format, ...)
{
    va_list ap;

    va_start(ap, format);
    int ret = vsprintf(s, format, ap);
    va_end(ap);

    return ret;
}

int __fprintf_chk(FILE* stream, int flag, const char* format, ...)
{
    va_list ap;

    va_start(ap, format);
    int ret = vfprintf(stream, format, ap);
    va_end(ap);

    return ret;
}