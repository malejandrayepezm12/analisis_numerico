import numpy as np

class OptimizadorNoLineal:

    @staticmethod
    def backtracking_armijo(f, grad_f, x, d, c=1e-4, rho=0.5, alpha_init=1.0):
        """
        Encuentra alpha que satisface la condición de Armijo.
        f, grad_f: funciones que reciben x y devuelven f(x), grad_f(x)
        x: punto actual
        d: dirección de descenso
        """
        alpha = alpha_init
        fx = f(x)
        grad_dot_d = grad_f(x) @ d  # gradiente(x_k)^T d_k

        while f(x + alpha*d) > fx + c*alpha*grad_dot_d:
            alpha *= rho

        return alpha

    @staticmethod
    def dir_gradiente(grad_f, x):
        "Dirección del gradiente"
        return -grad_f(x)

    @staticmethod
    def dir_newton(grad_f, hess_f, x):
        """Dirección de Newton"""
        H = hess_f(x)
        g = grad_f(x)
        return np.linalg.solve(H, -g)  

    @staticmethod
    def dir_bfgs(grad_f, x, H_approx):
        """Dirección BFGS."""
        return -H_approx @ grad_f(x)

    @staticmethod
    def actualizar_bfgs(H_k, s_k, y_k):
        """Actualiza la aproximación de la inversa del Hessiano."""
        rho_k = 1.0 / (y_k @ s_k)
        n = len(s_k)
        I = np.eye(n)
        term1 = I - rho_k * np.outer(s_k, y_k)
        term2 = I - rho_k * np.outer(y_k, s_k)
        H_kmas1 = term1 @ H_k @ term2 + rho_k * np.outer(s_k, s_k)
        return H_kmas1

    @staticmethod
    def optimizar(f, grad_f, x0, metodo, hess_f=None, tol=1e-6, max_iter=1000):
        """
        metodo: 'gradiente', 'newton', o 'bfgs'
        Devuelve: lista de puntos x_k, lista de f(x_k), lista de ||grad f(x_k)||
        """
        x = x0.copy()
        historial_x = [x.copy()]
        historial_f = [f(x)]
        historial_grad_norm = [np.linalg.norm(grad_f(x))]

        if metodo == 'bfgs':
            H_approx = np.eye(len(x))  # H_0 = I

        for k in range(max_iter):
            grad = grad_f(x)

            if np.linalg.norm(grad) < tol:
                break

            # Calcular dirección según el método
            if metodo == 'gradiente':
                d = OptimizadorNoLineal.dir_gradiente(grad_f, x)
            elif metodo == 'newton':
                d = OptimizadorNoLineal.dir_newton(grad_f, hess_f, x)
            elif metodo == 'bfgs':
                d = OptimizadorNoLineal.dir_bfgs(grad_f, x, H_approx)

            # Backtracking para elegir alpha
            alpha = OptimizadorNoLineal.backtracking_armijo(f, grad_f, x, d)

            # Actualizar x
            x_nuevo = x + alpha * d

            # Si es BFGS, actualizar la aproximación del Hessiano inverso
            if metodo == 'bfgs':
                s_k = x_nuevo - x
                y_k = grad_f(x_nuevo) - grad
                if y_k @ s_k > 1e-10:  # evitar división por 0 
                    H_approx = OptimizadorNoLineal.actualizar_bfgs(H_approx, s_k, y_k)

            x = x_nuevo
            historial_x.append(x.copy())
            historial_f.append(f(x))
            historial_grad_norm.append(np.linalg.norm(grad_f(x)))

        return np.array(historial_x), np.array(historial_f), np.array(historial_grad_norm)