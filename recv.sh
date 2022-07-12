if [ $1 -eq 0 ]; then
	rsync -azvhP --exclude={'Jumper-v0_PlatformJumper-v0'} katayama@server1:~/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
	rsync -azvhP --exclude={'Jumper-v0_PlatformJumper-v0'} katayama@server3:~/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
	rsync -azvhP --exclude={'Jumper-v0_PlatformJumper-v0'} katayama@server4:~/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
	rsync -azvhP --exclude={'Jumper-v0_PlatformJumper-v0'} katayama@server5:~/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
	rsync -azvhP u00576@ist_cluster:/home/u00576/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
elif [ "$1" = "ist" ]; then
	rsync -azvhP u00576@ist_cluster:/home/u00576/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
else
	rsync -azvhP -e "ssh -i ~/.ssh/id_ed25519" --exclude={'Jumper-v0_PlatformJumper-v0'} katayama@server$1:~/workspace/evogym/examples/saved_data/ ~/katayama/evogym/examples/saved_data/
fi
