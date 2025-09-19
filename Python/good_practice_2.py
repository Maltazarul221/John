import good_practice

# cand import un fisier in altul, DocString-ul sau este accesibil prin atributul .__doc__
print(good_practice.__doc__)


def main():
    print("Test add:", good_practice.add(2, 3))

if __name__ == "__main__":
    main()