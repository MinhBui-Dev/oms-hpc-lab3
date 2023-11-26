#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "terarec.h"

typedef struct partition_struct{
	terarec_t* data;
	int size;
} partition_t;

terarec_t* select_samples(terarec_t* local_data, int local_len, int sample_size){
	terarec_t* samples = malloc(sizeof(terarec_t) * sample_size);
	int step = local_len / (sample_size+1);
	for(int i = 0; i < sample_size; i++)
		samples[i] = local_data[(i+1)*step];
	return samples;
}

terarec_t* select_splitters(terarec_t* samples, int sample_size, int splitter_size){
	terarec_t* splitters = malloc(sizeof(terarec_t) * splitter_size);
	int step = sample_size / (splitter_size+1);
	for(int i = 0; i < splitter_size; i++)
		splitters[i] = samples[(i+1)*step];
	return splitters;
}

int find_index_less_than(terarec_t* data, int left, int right, terarec_t* splitters){
	//printf("left: %d, right: %d\n", left, right);
	int result = -1;
	while (left <= right)
	{
		int mid = (left + right) / 2;
		if (teraCompare(&data[mid], splitters) < 0)
		{
			result = mid;
			left = mid + 1;
		} else {
			right = mid - 1;
		}
	}
	return result;
}

void merge(terarec_t* data, int n, terarec_t* temp){
	int i = 0;
	int j = n/2;
	int k = 0;
	while (i < n/2 && j < n)
	{
		if (teraCompare(&data[i], &data[j]) < 0)
		{
			temp[k] = data[i];
			i++;
		} else {
			temp[k] = data[j];
			j++;
		}
		k++;
	}
	while (i < n/2)
	{
		temp[k] = data[i];
		i++;
		k++;
	}
	while (j < n)
	{
		temp[k] = data[j];
		j++;
		k++;
	}
	memcpy(data, temp, sizeof(terarec_t) * n);
}

void merge_sort(terarec_t* data, int n, terarec_t* temp){
	if (n < 2)
	{
		return;
	}
	#pragma omp task shared(data) if(n > 100)
	merge_sort(data, n/2, temp);

	#pragma omp task shared(data) if(n > 100)
	merge_sort(data + n/2, n - n/2, temp);

	#pragma omp taskwait
	merge(data, n, temp);
}
void terasort(terarec_t *local_data, int  local_len, 
			  terarec_t **sorted_data, int* sorted_counts, long* sorted_displs){
	
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
	MPI_Comm_size (MPI_COMM_WORLD, &P);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	terarec_t* temp = malloc(sizeof(terarec_t) * local_len);


	// 1. Local Sort

	// merge sort
	#pragma omp parallel
	{
		#pragma omp single
		merge_sort(local_data, local_len, temp);
	}
	//qsort(local_data, local_len, sizeof(terarec_t), teraCompare);

	free(temp);
	// 2. Select P-1 sample
	int sample_size = P-1;
	terarec_t* samples = select_samples(local_data, local_len, sample_size);

	// 3. Gather samples to root
	terarec_t* all_samples = malloc(sizeof(terarec_t) * sample_size * P);

	//printf("rank %d, sample size %d\n", rank, sample_size);
	
	MPI_Gather(samples, sample_size, mpi_tera_type, 
			   all_samples, sample_size, mpi_tera_type, 
			   0, MPI_COMM_WORLD);

	// 4. Sort samples on root
	
	terarec_t* splitters = NULL;
	int splitter_size = P-1;

	if (rank == 0)
	{
		// sort samples
		terarec_t* temp = malloc(sizeof(terarec_t) * sample_size * P);
		// merge sort
		#pragma omp parallel
		{
			#pragma omp single
			merge_sort(all_samples, sample_size * P, temp);
		}

		free(temp);
		//qsort(all_samples, sample_size * P, sizeof(terarec_t), teraCompare);

		// print sorted samples
		// printf("sorted samples:\n");
		// for (int i = 0; i < sample_size * P; i++)
		// {
		// 	printf("%d: %s\n", i, all_samples[i].value);
		// }

		// 5. Select P-1 splitters
		splitters = select_splitters(all_samples, sample_size * P, splitter_size);
	} else {
		splitters = malloc(sizeof(terarec_t) * splitter_size);
	}

	// print splitters
	// if (rank == 0)
	// {
	// 	printf("splitters:\n");
	// 	for (int i = 0; i < splitter_size; i++)
	// 	{
	// 		printf("%d: %s\n", i, splitters[i].value);
	// 	}
	// }
	
	
	//printf("Local len: %d\n", local_len);



	free(all_samples);
	free(samples);

	

	// 6. Broadcast Splitters
	MPI_Bcast(splitters, splitter_size, mpi_tera_type, 0, MPI_COMM_WORLD);

	// 7. Partition Using Splitters
	// sample the size of each partition by loop through local data and compare with splitters
	partition_t* partitions = malloc(sizeof(partition_t) * P);

	// initialize partition size
	for (int i = 0; i < P; i++)
	{
		partitions[i].size = 0;
	}
	int splitter_index = 0;

	// find partition size
	int partition_index = 0;
	int left = 0;
	int right = local_len - 1;
	for (partition_index = 0; partition_index < P-1; partition_index++)
	{
		int end_index = find_index_less_than(local_data, left, right, &splitters[splitter_index]);
		if (end_index != -1){
			partitions[partition_index].size = end_index - left + 1;
			left = end_index + 1;
		} else {
			partitions[partition_index].size = 0;
		}		
		splitter_index++;
	}

	// find last partition size
	partitions[partition_index].size = local_len - left;

	// print partition size regardless of rank
	// printf("partition size of rank %d: ", rank);
	// for (int i = 0; i < P; i++)
	// {
	// 	printf("%d ", partitions[i].size);
	// }
	// printf("\n");

	
	
	// allocate memory for each partition
	for (int i = 0; i < P; i++)
	{
		partitions[i].data = calloc(partitions[i].size, sizeof(terarec_t));
	}

	// move data to each partition using memcpy
	int data_index = 0;
	for (int i = 0; i < P; i++)
	{
		if (partitions[i].size != 0)
		{
			memcpy(partitions[i].data, local_data + data_index, sizeof(terarec_t) * partitions[i].size);
			data_index += partitions[i].size;
		}
	}
	


	free(splitters);
	// 8. Exchange values
	// each partition number corresponds to a rank number of process, i.e. partition 0 -> rank 0
	// employ cyclic shift to exchange values
	// for example, rank 0 send to rank 1, rank 1 send to rank 2, ..., rank P-1 send to rank 0
	// while rank 0 receive from rank P-1, rank 1 receive from rank 0, ..., rank P-1 receive from rank P-2
	// then in the next iteration, rank 0 send to rank 2, rank 1 send to rank 3, ..., rank P-1 send to rank 1
	// while rank 0 receive from rank P-2, rank 1 receive from rank P-1, ..., rank P-1 receive from rank P-3

	// allocate buffer for receiving data
	terarec_t* recv_buff = malloc(sizeof(terarec_t) * local_len);
	terarec_t* recv_buff_2 = malloc(sizeof(terarec_t) * local_len);
	partition_t* recv_partition = malloc(sizeof(partition_t) * P);

	
	MPI_Status old_status;

	for (int i = 1; i < P; i++)
	{
		int dest = (rank + i) % P;
		int source = (rank - i + P) % P;
		MPI_Status receive_status;
		MPI_Request send_request, receive_request;
		MPI_Isend(partitions[dest].data, partitions[dest].size, mpi_tera_type, dest, 0, MPI_COMM_WORLD, &send_request);
		MPI_Irecv(recv_buff, local_len, mpi_tera_type, source, 0, MPI_COMM_WORLD, &receive_request);
		if (i > 1)
		{
			int count = 0;
			MPI_Get_count(&old_status, mpi_tera_type, &count);
			recv_partition[old_status.MPI_SOURCE].data = malloc(sizeof(terarec_t) * count);
			memcpy(recv_partition[old_status.MPI_SOURCE].data, recv_buff_2, sizeof(terarec_t) * count);
			recv_partition[old_status.MPI_SOURCE].size = count;
		}
		
		MPI_Wait(&receive_request, &receive_status);
		old_status = receive_status;
		terarec_t* temp = recv_buff_2;
		recv_buff_2 = recv_buff;
		recv_buff = temp;
	}

	// receive data from last rank
	int count = 0;
	MPI_Get_count(&old_status, mpi_tera_type, &count);
	recv_partition[old_status.MPI_SOURCE].data = malloc(sizeof(terarec_t) * count);
	memcpy(recv_partition[old_status.MPI_SOURCE].data, recv_buff_2, sizeof(terarec_t) * count);
	recv_partition[old_status.MPI_SOURCE].size = count;


	free(recv_buff);

	// copy data from own partition to recv_partition
	recv_partition[rank].data = malloc(sizeof(terarec_t) * partitions[rank].size);
	memcpy(recv_partition[rank].data, partitions[rank].data, sizeof(terarec_t) * partitions[rank].size);
	recv_partition[rank].size = partitions[rank].size;

	// find total size of unsorted data
	int total_size = 0;

	for (int i = 0; i < P; i++)
	{
		total_size += recv_partition[i].size;
	}

	// allocate memory for unsorted data
	*sorted_data = malloc(sizeof(terarec_t) * total_size);

	// copy data from recv_partition to unsorted_data
	int index = 0;
	for (int i = 0; i < P; i++)
	{
		memcpy(*sorted_data + index, recv_partition[i].data, sizeof(terarec_t) * recv_partition[i].size);
		index += recv_partition[i].size;
	}

	// sort unsorted data
	// merge sort
	temp = malloc(sizeof(terarec_t) * total_size);
	#pragma omp parallel
	{
		#pragma omp single
		merge_sort(*sorted_data, total_size, temp);
	}
	
	free(temp);
	//qsort(*sorted_data, total_size, sizeof(terarec_t), teraCompare);

	//print sorted counts
	//printf("sorted counts of rank %d: %d\n", rank, total_size);

	// Allgather total size of sorted data to root
	MPI_Allgather(&total_size, 1, MPI_INT, sorted_counts, 1, MPI_INT, MPI_COMM_WORLD);

	// calculate displacements
	sorted_displs[0] = 0;
	for (int i = 1; i < P; i++)
	{
		sorted_displs[i] = sorted_displs[i-1] + sorted_counts[i-1];
	}

	// free partitions data
	for (int i = 0; i < P; i++)
	{
		free(partitions[i].data);
		free(recv_partition[i].data);
	}

	free(partitions);
	free(recv_partition);



	/* terarec_t* all_data = NULL;

	if(rank == 0){
		all_data = (terarec_t*) malloc(sizeof(terarec_t) * P * local_len);
	}

	MPI_Gather(local_data, local_len, mpi_tera_type,
			   all_data, local_len, mpi_tera_type, 0, MPI_COMM_WORLD);

	if(rank == 0){
		qsort(all_data, P * local_len, sizeof(terarec_t), teraCompare);
	}

	*sorted_data = malloc(sizeof(terarec_t) * local_len);

	MPI_Scatter(all_data, local_len, mpi_tera_type, 
				*sorted_data, local_len, mpi_tera_type, 
				0, MPI_COMM_WORLD);

	for(int i = 0; i < P; i++)
		sorted_counts[i] = local_len;
	sorted_displs[0] = 0;
	for(int i = 1; i < P; i++)
		sorted_displs[i] = sorted_displs[i-1] + sorted_counts[i-1];

	if(rank == 0)
		free(all_data); */
}
