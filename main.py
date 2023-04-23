from graph import Graph

if __name__ == "__main__":
    graph = Graph(weighted=True, directed=True)
    graph.read_from_file("graph.txt")
