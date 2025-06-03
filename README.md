# Master-Thesis
The code uses to wirite the thesis "Flow Tracing and Nodal Cost Allocation In Large-N Renewable Electricity Networks". The thesis models the European Electric Power Grid both as a Large-N and a Small-N network, to enable comparison, of various cost allocation schemes used to determine nodal and country costs, including a flow tracing algortihm.

USE:
To use this code run Main2 script. This Yields the reuslts for the Large-N network.
To obtain results for Small-N network, run the Smalln Script subsequent to the Large-N script.

Various forms of these scripts has been used throughout the witing project, most using parralelization on various levels.

Depending on Computational power and choice of layouts etc. it will take around 6-8 hours on a Regular pc. Leaving out the GAS-Layout will reduce this significantly.
Consider using parralelizing over the various layouts. This reduces the running time significantly.
The GAS-Script utilizes internal Parralelization, which can cause issues if parralelization is also used in the upper levels. Pseudo parralelization,i.e. through multiple consoles, can be an option, but sometimes causes memory allocation issues. 32Gb memory was originally shown to be enough.
In future versions parralelization should be performed inside the flowtracing script, as this is the main driver of computational costs. This would also allow the GAS layout parralization to remain. Alternative hardware use, such as GPU utilization, should also be considered.

DATA:
The thesis was written using hourly data from 2013, provided by M. Beltoft. This data has not been uploaded due to file-size constraints, but can be provided by request. 
The code should largely be able to run for other datasets provided they are formatted properly. Alternatively, change the 'data_loader' script to translate your data.
If using a differnt number of timesteps, i.e. a different length of dataset, it should be changed in the input parameters 'n_t'. Note that if fewer timesteps are used, it might cuase issues with the flowtracing algortihm. This can be fixed by changing the bin size, but this in turn causes instability.
