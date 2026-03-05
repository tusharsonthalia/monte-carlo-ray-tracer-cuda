.DEFAULT_GOAL := all

# ---------- Compilers ----------
CC = gcc
NVCC = nvcc

# ---------- Directories ----------
IDIR = include
SDIR = src
ODIR = obj

# ---------- Dependencies ----------
_DEPS = ray_tracing.h timer.h rng.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

# ---------- Common Flags ----------
CFLAGS_COMMON = -O3 -Wall -Wextra -I$(IDIR)

# ---------- Objects ----------
TIMER_OBJ = $(ODIR)/timer.o

# ---------- Create object directory ----------
$(ODIR):
	mkdir -p $(ODIR)

# ---------- Timer (shared) ----------
$(ODIR)/timer.o: $(SDIR)/timer.c $(DEPS) | $(ODIR)
	$(CC) $(CFLAGS_COMMON) -c -o $@ $<
