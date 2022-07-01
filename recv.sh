if [ $1 -eq 0 ]; then
	rsync -azvhP katayama@server1:~/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
	rsync -azvhP katayama@server3:~/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
	rsync -azvhP katayama@server4:~/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
	rsync -azvhP katayama@server5:~/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
else
	rsync -azvhP katayama@server$1:~/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
fi
