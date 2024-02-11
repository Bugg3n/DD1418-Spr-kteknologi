#! /bin/sh
python random_indexing.py -c -co cleaned_example.txt
diff --strip-trailing-cr correct_cleaned_example.txt cleaned_example.txt
error=$?
if [ $error -eq 0 ]
then
	echo "Success!"
else
	echo "Some problems were found"
fi
