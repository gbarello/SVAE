
class log:
    def __init__(self,f,name = ""):
        self.FNAME = f

        F = open(self.FNAME,"w")
        if len(name) != 0:
            F.write(name + "\n")
        F.close()

    def log(self,data,PRINT = True):

        if PRINT:
            print(data)

        F = open(self.FNAME,"a")
        F.write(str(data[0]))
        for k in data[1:]:
            F.write(",{}".format(k))
        F.write("\n")
        F.close()

