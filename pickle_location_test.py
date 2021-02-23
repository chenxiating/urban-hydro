import pickle
import datetime as date
import datetime
import time
import os

def main(process_core_name=None):
	today = date.datetime.today()
	dt_str = today.strftime("%Y%m%d-%H%M")
#	file_directory = os.path.dirname(os.path.abspath(__file__))
#	datafile_directory=file_directory +'/datafiles_'+dt_str
#	print('os.path.exists(datafile_directory)', os.path.exists(datafile_directory))
#	if not os.path.exists(datafile_directory):
#	    os.makedirs(datafile_directory)
#	os.chdir(datafile_directory)
	for k in range(10):
		datafile_name = 'file_'+dt_str+'_core-'+str(process_core_name)+'k_'+str(k)
		f = open(datafile_name,'wb')
		file_location = os.path.dirname(os.path.abspath(datafile_name))
		print('File location:',file_location)
		#datafile_directory=file_directory +'/datafiles_'+dt_str
		output_df = [12345,678]
		pickle.dump(output_df, f)
		f.close()
		print("File name is: ", datafile_name, "File size: ", os.path.getsize(datafile_name), "Total time: ")

if __name__ == '__main__':
    main()
