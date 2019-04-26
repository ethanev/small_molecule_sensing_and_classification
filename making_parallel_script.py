# ok the purpose of this script is to generate all the possible selector combinations. without duplicates
import itertools
import argparse

def make_sh_file(selector_group, number, sel_num):
	f = open('./launch_model_train_{}.sh'.format(number),'w')
	select = str(selector_group).replace(' ','')
	f.write('python ./parallel_opt_selectors.py -g {} -t {} -n {}'.format(select,number,sel_num))
	f.close()

def make_master_file(number):
	f = open('./launch_all.sh', 'w')
	for i in range(number):
		f.write('qsub -V -cwd ./launch_model_train_{}.sh'.format(i))
		f.write('\n')
	f.close()


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', help='number of selectors in use')
	args = parser.parse_args()
	sel_number = int(args.n)
	if sel_number == 4:
		number = 404
	elif sel_number == 3:
		number = 95
	elif sel_number == 5:
		number = 1292
	return number, sel_number

def main():
	number, sel_number = parse_arguments()
	make_master_file(number)
	selectors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	combos = list(itertools.combinations(selectors,sel_number))
	combos = [list(combo) for combo in combos]
	print(combos)
	i = 0
	for j in range(number):
		block = combos[i:i+12]
		i += 12
		make_sh_file(block, j, sel_number)
		

if __name__ == '__main__':
	main()
