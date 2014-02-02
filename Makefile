UNAME   := $(shell uname)
UNAME_P := $(shell uname -p)

# Defaults
CC = cc
CPP = c++

ARCH_386 =
ARCH_x64 =
ARCH_FLAGS =
LINKFLAGS = 

# Mac OS X
ifeq ($(UNAME),Darwin)
CC = clang
endif

# Linux
ifeq ($(UNAME),Linux)
LINKFLAGS = -pthread
endif

# SSE
ifeq ($(UNAME_P),i386)
ARCH_386 = -arch i386
ARCH_x64 = -arch x86_64
ARCH_FLAGS = -msse2 -DLIBDIVIDE_USE_SSE2=1
endif
ifeq ($(UNAME_P),x86_64)
ARCH_386 = -arch i386
ARCH_x64 = -arch x86_64
ARCH_FLAGS = -msse2 -DLIBDIVIDE_USE_SSE2=1
endif
ifeq ($(UNAME_P),amd64)
ARCH_386 = -arch i386
ARCH_x64 = -arch x86_64
ARCH_FLAGS = -msse2 -DLIBDIVIDE_USE_SSE2=1
endif

# NEON
ifeq ($(UNAME_P),armv7l)
ARCH_FLAGS = -mfpu=neon -DLIBDIVIDE_USE_NEON=1
endif

DEBUG_FLAGS   = -fstrict-aliasing -W -Wall -g -O0 -DLIBDIVIDE_ASSERTIONS_ON=1 $(ARCH_FLAGS) $(LINKFLAGS)
RELEASE_FLAGS = -fstrict-aliasing -W -Wall -g -O3 $(ARCH_FLAGS) $(LINKFLAGS)

tester: debug
	

debug: libdivide_test.cpp libdivide.h
	$(CPP) $(DEBUG_FLAGS) $(ARCH_386) $(ARCH_x64) -g -o tester libdivide_test.cpp

i386: libdivide_test.cpp libdivide.h
	$(CPP) $(DEBUG_FLAGS) $(ARCH_386) -o tester libdivide_test.cpp

x86_64: libdivide_test.cpp libdivide.h
	$(CPP) $(DEBUG_FLAGS) $(ARCH_x64) -o tester libdivide_test.cpp

release: libdivide_test.cpp libdivide.h
	$(CPP) $(RELEASE_FLAGS) $(ARCH_x64) $(ARCH_386) -o tester libdivide_test.cpp

benchmark: libdivide_benchmark.c libdivide.h
	$(CC) $(RELEASE_FLAGS) $(ARCH_x64) $(ARCH_386) -o benchmark libdivide_benchmark.c

clean:
	rm -Rf tester tester.dSYM benchmark benchmark.dSYM

install:
	@echo "libdivide does not install! Just copy the header libdivide.h into your projects."
