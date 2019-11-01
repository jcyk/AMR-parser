for file in `ls *test_out`;
do
    if test -f $file
    then
        echo $file
        python smatch/smatch.py --pr -f $file ../preprocessing/2017/test.txt_processed_preprocess
    else
        echo $file "is dir"
    fi
done
