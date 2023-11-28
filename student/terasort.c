#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "terarec.h"
#define val_name(a) #a

typedef struct partition_struct
{
	terarec_t *data;
	int size;
} partition_t;

terarec_t *select_samples(terarec_t *local_data, int local_len, int sample_size)
{
	terarec_t *samples = malloc(sizeof(terarec_t) * sample_size);
	int step = local_len / (sample_size + 1);
	for (int i = 0; i < sample_size; i++)
		samples[i] = local_data[(i + 1) * step];
	return samples;
}

void select_splitters(terarec_t *samples, terarec_t *splitters, int sample_size, int splitter_size)
{
	int step = sample_size / (splitter_size + 1);
	for (int i = 0; i < splitter_size; i++)
		splitters[i] = samples[(i + 1) * step];
}

int find_index_less_than(terarec_t *data, int left, int right, terarec_t *splitters)
{
	int result = -1;
	while (left <= right)
	{
		int mid = (left + right) / 2;
		if (teraCompare(&data[mid], splitters) < 0)
		{
			result = mid;
			left = mid + 1;
		}
		else
		{
			right = mid - 1;
		}
	}
	return result;
}

void debug_print_array(int *array, int size, char *variable, int rank)
{
	fprintf(stderr, "rank %d, %s: ", rank, variable);
	for (int i = 0; i < size; i++)
	{
		fprintf(stderr, "%d ", array[i]);
	}
	fprintf(stderr, "\n");
}

void terasort(terarec_t *local_data, int local_len,
			  terarec_t **sorted_data, int *sorted_counts, long *sorted_displs)
{

	/*
	1. Local Sort
	2. Select P-1 sample
	3. Gather samples to root
	4. Sort samples on root
	5. Select P-1 splitters
	6. Broadcast Splitters
	7. Partition Using Splitters
	8. Exchange values
	*/
	

	int rank, P;
	MPI_Comm_size(MPI_COMM_WORLD, &P);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	qsort(local_data, local_len, sizeof(terarec_t), teraCompare);

	// 2. Select P-1 sample
	int sample_size = P - 1;
	terarec_t *samples = select_samples(local_data, local_len, sample_size);

	// 3. Gather samples to root
	terarec_t *all_samples = malloc(sizeof(terarec_t) * sample_size * P);

	MPI_Gather(samples, sample_size, mpi_tera_type,
			   all_samples, sample_size, mpi_tera_type,
			   0, MPI_COMM_WORLD);

	// 4. Sort samples on root
	int splitter_size = P - 1;
	terarec_t *splitters = calloc(splitter_size, sizeof(terarec_t));

	if (rank == 0)
	{
		qsort(all_samples, sample_size * P, sizeof(terarec_t), teraCompare);
		// 5. Select P-1 splitters
		select_splitters(all_samples, splitters, sample_size * P, splitter_size);
	}
	
	// 6. Broadcast Splitters

	MPI_Bcast(splitters, splitter_size, mpi_tera_type, 0, MPI_COMM_WORLD);

	// 7. Partition Using Splitters
	int *partition_sizes = malloc(sizeof(partition_t) * P);
	int *partition_displs = malloc(sizeof(partition_t) * P);

	// find partition size
	int splitter_index = 0;
	int partition_index = 0;
	int left = 0;
	int right = local_len - 1;
	for (partition_index = 0; partition_index < P - 1; partition_index++)
	{
		int end_index = find_index_less_than(local_data, left, right, &splitters[splitter_index]);
		if (end_index != -1)
		{
			partition_displs[partition_index] = left;
			partition_sizes[partition_index] = end_index - left + 1;
			left = end_index + 1;
		}
		else
		{
			partition_sizes[partition_index] = 0;
			partition_displs[partition_index] = left;
		}
		splitter_index++;
	}

	// find last partition size
	partition_sizes[partition_index] = local_len - left;
	partition_displs[partition_index] = left;

	// debug_print_array(partition_displs, P, val_name(partition_displs), rank);
	// debug_print_array(partition_sizes, P, val_name(partition_sizes), rank);

	// 8. Exchange values

	// allocate buffer for receiving data
	int *recv_partition_sizes = malloc(sizeof(int) * P);
	int *recv_partition_displs = malloc(sizeof(int) * P);

	MPI_Alltoall(partition_sizes, 1, MPI_INT, recv_partition_sizes, 1, MPI_INT, MPI_COMM_WORLD);

	// debug_print_array(recv_partition_sizes, P, val_name(recv_partition_sizes), rank);

	// calculate displacements and total size of recv data
	int total_size = 0;
	recv_partition_displs[0] = 0;

	for (size_t i = 0; i < P; i++)
	{
		total_size += recv_partition_sizes[i];

		if (i > 0)
		{
			recv_partition_displs[i] = recv_partition_displs[i - 1] + recv_partition_sizes[i - 1];
		}
	}

	// allocate buffer for unsorted data
	*sorted_data = malloc(sizeof(terarec_t) * total_size);

	MPI_Alltoallv(local_data, partition_sizes, partition_displs, mpi_tera_type,
				  *sorted_data, recv_partition_sizes, recv_partition_displs, mpi_tera_type,
				  MPI_COMM_WORLD);

	
	
	// Allgather total size of sorted data to root

	// double start = MPI_Wtime();
	// MPI_Request request;
	// MPI_Iallgather(&total_size, 1, MPI_INT, sorted_counts, 1, MPI_INT, MPI_COMM_WORLD, &request);
	MPI_Allgather(&total_size, 1, MPI_INT, sorted_counts, 1, MPI_INT, MPI_COMM_WORLD);

	qsort(*sorted_data, total_size, sizeof(terarec_t), teraCompare);
	
	// MPI_Wait(&request, MPI_STATUS_IGNORE);
	// double end = MPI_Wtime();
	// fprintf(stderr,"rank %d, time: %f\n", rank, end - start);

	// calculate displacements
	sorted_displs[0] = 0;
	int cummulative = 0;
	int i = 1;
	for (i = 1; i < P; i++)
	{
		cummulative += sorted_counts[i - 1];
		sorted_displs[i] = cummulative;
	}
\

	free(partition_sizes);
	free(partition_displs);
	free(recv_partition_sizes);
	free(recv_partition_displs);
	
	free(all_samples);
	free(samples);
	free(splitters);
}
