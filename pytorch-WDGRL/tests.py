class hah:
    def __init__(self):
        super(hah, self).__init__()
        self.ha=5

g=hah()

aha=g.ha

g.ha=6
print(aha)
print(g.ha)
