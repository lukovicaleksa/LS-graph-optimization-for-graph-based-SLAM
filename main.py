from pg_classes import * 

if __name__ == '__main__':
    # Pose Graph creatin
    pg = PoseGraph()

    # Read Pose Graph from files
    pg.read_graph(v_data = 'Data/Vertices.dat', e_data = 'Data/Edges.dat')

    # Plot Unoptimized Pose Graph
    plt.figure(1)
    pg.plot_graph(title = 'Unoptimized Graph', show_constraints = False)

    # Plot Unoptimized Pose Graph with constraints
    plt.figure(2)
    pg.plot_graph(title = 'Unoptimized Graph with Constraints', show_constraints = True)

    # Graph Optimization
    norm_dX = pg.optimize(max_iterations = 20, tolerance = 0.1)

    # Plot Graph Convergence
    plt.figure(3)
    plt.plot(norm_dX, color = 'green')
    plt.title('Pose Graph updates per iteration')
    plt.ylabel('|dX|')
    plt.xlabel('iteration')
    plt.grid()

    # Plot Optimized Graph
    plt.figure(4)
    pg.plot_graph(title = 'Optimized Graph', show_constraints = True)

    plt.show()