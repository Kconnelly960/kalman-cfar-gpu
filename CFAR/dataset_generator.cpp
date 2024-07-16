
#include "wb.h"

static char *base_dir;
const size_t NUM_BINS      = 4096;
const unsigned int BIN_CAP = 127;

static void compute(unsigned int *bins, unsigned int *input, int num) {
  for (int i = 0; i < num; ++i) {
    int idx = input[i];
    if (bins[idx] < BIN_CAP)
      ++bins[idx];
  }
}

static unsigned int *generate_data(size_t n, unsigned int num_bins) {
  unsigned int *data = (unsigned int *)malloc(sizeof(unsigned int) * n);
  for (unsigned int i = 0; i < n; i++) {
    data[i] = rand() % num_bins;
	//data[i] = 0;
  }
  return data;
}

static void write_data(char *file_name, unsigned int *data, int num) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d", num);
  for (int ii = 0; ii < num; ii++) {
    fprintf(handle, "\n%d", *data++);
  }
  fflush(handle);
  fclose(handle);
}

static void create_dataset(int datasetNum, size_t input_length,
                           size_t num_bins) {

  const char *dir_name =
      wbDirectory_create(wbPath_join(base_dir, datasetNum));

  char *input_file_name  = wbPath_join(dir_name, "input.raw");
  char *output_file_name = wbPath_join(dir_name, "output.raw");

  unsigned int *input_data = generate_data(input_length, num_bins);
  unsigned int *output_data =
      (unsigned int *)calloc(sizeof(unsigned int), num_bins);

  compute(output_data, input_data, input_length);

  write_data(input_file_name, input_data, input_length);
  write_data(output_file_name, output_data, num_bins);

  free(input_data);
  free(output_data);
}

int main() {
  base_dir = wbPath_join(wbDirectory_current(), "Histogram", "Dataset");

  create_dataset(0, 8192, NUM_BINS);
  create_dataset(1, 4098, NUM_BINS);
  create_dataset(2, 12000, NUM_BINS);
  create_dataset(3, 5000, NUM_BINS);
  create_dataset(4, 30240, NUM_BINS);
  create_dataset(5, 100000, NUM_BINS);
  create_dataset(6, 500000, NUM_BINS);
  return 0;
}
