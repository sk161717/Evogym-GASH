if [ "$1" = "ist" ]; then
	rsync -azvhP --exclude={'.git','examples/saved_data','build','examples/__pycache__','examples/pyME/map_elites/__pycache__/'} ~/katayama/evogym/ u00576@ist_cluster:/home/u00576/workspace/evogym/
else
	rsync -azvhP --exclude={'.git','examples/saved_data','build','examples/__pycache__','examples/pyME/map_elites/__pycache__/'} ~/katayama/evogym/ server$1:~/workspace/evogym/
	ssh server$1 '/home/katayama/.pyenv/shims/python /home/katayama/workspace/evogym/setup.py install'
fi
 
