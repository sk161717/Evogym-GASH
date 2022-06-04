class B():
    def __init__(self) -> None:
        self.b=0
    
    def add(self):
        self.b+=1

b=B()

def foo():
    b.add()
    b=B()
    return 

def main():
    foo()
    foo()
    print(b.b)

main()