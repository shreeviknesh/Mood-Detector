from MoodDetector import train, detect
import sys

USAGE = "[Usage] python App.py [[-t/-T/train]/[-d/-D/detect]]"

if __name__ == "__main__":
	argv = sys.argv
	argc = len(argv)

	if argc != 2:
		print("[Error] Invalid number of Arguments")
		print(USAGE)
		exit(0)

	option = argv[1]
	if option == "-t" or option == "-T" or option == "train":
		train()
	elif option == "-d" or option == "-D" or option == "detect":
		detect()
	else:
		print("[Error] Invalid Arguments")
		print(USAGE)