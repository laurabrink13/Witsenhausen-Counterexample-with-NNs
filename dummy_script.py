import sys


first_arg = sys.argv[1]
second_arg = sys.argv[2]

def greetings(word1=first_arg, word2=second_arg):
    print("{} {}".format(word1, word2))

if __name__ == "__main__":
	print("{} and also {}".format(first_arg, second_arg))
    # greetings()
    # i = 0
    # for argument in sys.argv: 
    # 	print('argument {} is {}'.format(i, argument))
    # 	i += 1
    # # greetings("Bonjour", "monde")