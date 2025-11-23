import math, random, numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class MCTSNode:
    __slots__ = ("padre","hijos","visitas","valor","accion_padre","estado","expandido","bono")
    def __init__(self, pad=None, acc=None, est=None):
        self.padre = pad
        self.hijos = {}
        self.visitas = 0
        self.valor = 0.0
        self.accion_padre = acc
        self.estado = est
        self.expandido = False
        self.bono = 0.0

    def uct_score(self, pv, c):
        return float("inf") if self.visitas == 0 else (self.valor + self.bono) / self.visitas + c * math.sqrt(math.log(max(1, pv)) / self.visitas)

class MCTSAgent(Policy):
    def __init__(self, it=100, c=1.2, rl=20, seed=None):
        self.iteraciones = it
        self.c = c
        self.limite_rollout = rl
        self.rng = random.Random(seed)
        self.tabla_valores = {}
        self.iter_minima = 50
        self.transiciones = {}
        self.recompensas = {}
        self.valores_v = {}

    def mount(self, *args, **kwargs):
        pass

    def _id_estado(self, estado):
        return estado.board.tobytes(), estado.player

    def _actualizar_adp(self, estado_ant, accion, estado_sig):
        id_ant = self._id_estado(estado_ant)
        id_sig = self._id_estado(estado_sig)
        if id_ant not in self.transiciones:
            self.transiciones[id_ant] = {}
        if accion not in self.transiciones[id_ant]:
            self.transiciones[id_ant][accion] = {}
        if id_sig not in self.transiciones[id_ant][accion]:
            self.transiciones[id_ant][accion][id_sig] = 0
        self.transiciones[id_ant][accion][id_sig] += 1
        if id_ant not in self.recompensas:
            self.recompensas[id_ant] = {}
        self.recompensas[id_ant][accion] = estado_sig.get_winner()

    def _v_adp(self, estado):
        id_e = self._id_estado(estado)
        if id_e in self.valores_v:
            return self.valores_v[id_e]
        if id_e not in self.transiciones:
            return 0
        mejor = -1e18
        for accion in self.transiciones[id_e]:
            n_total = sum(self.transiciones[id_e][accion].values())
            suma = 0
            for id_sig, n in self.transiciones[id_e][accion].items():
                p = n / n_total
                jugador_sig = id_sig[1]
                b = np.frombuffer(id_sig[0], dtype=np.int64).reshape(6,7)
                est_sig = ConnectState(b, jugador_sig)
                r = 0
                if id_e in self.recompensas and accion in self.recompensas[id_e]:
                    r = self.recompensas[id_e][accion]
                suma += p * (r + self._v_adp(est_sig))
            if suma > mejor:
                mejor = suma
        self.valores_v[id_e] = mejor
        return mejor

    def _ow(self, estado, oponente):
        tablero = estado.board
        filas = estado.ROWS
        columnas = estado.COLS
        libres = estado.get_free_cols()
        for col in libres:
            try:
                nuevo_tablero = estado.transition(col).board
            except:
                continue
            for fila in range(filas):
                fila_arr = nuevo_tablero[fila]
                for col2 in range(columnas-3):
                    ventana = fila_arr[col2:col2+4]
                    if np.sum(ventana == oponente) == 2 and np.sum(ventana == 0) == 2:
                        return True
            for col2 in range(columnas):
                col_arr = nuevo_tablero[:, col2]
                for fila2 in range(filas-3):
                    ventana = col_arr[fila2:fila2+4]
                    if np.sum(ventana == oponente) == 2 and np.sum(ventana == 0) == 2:
                        return True
            for fila2 in range(filas-3):
                for col2 in range(columnas-3):
                    diag1 = np.array([nuevo_tablero[fila2+i, col2+i] for i in range(4)])
                    if np.sum(diag1 == oponente) == 2 and np.sum(diag1 == 0) == 2:
                        return True
                    diag2 = np.array([nuevo_tablero[fila2+i, col2+3-i] for i in range(4)])
                    if np.sum(diag2 == oponente) == 2 and np.sum(diag2 == 0) == 2:
                        return True
        return False

    def _h(self, estado):
        llave = estado.board.tobytes()
        if llave in self.tabla_valores:
            return self.tabla_valores[llave]
        tablero = estado.board
        filas = estado.ROWS
        columnas = estado.COLS
        puntaje = 0
        def sw(ventana):
            negativos = np.sum(ventana == -1)
            positivos = np.sum(ventana == 1)
            espacios = np.sum(ventana == 0)
            s = 0
            if negativos == 3 and espacios == 1: s += 60
            if negativos == 2 and espacios == 2: s += 15
            if positivos == 3 and espacios == 1: s -= 80
            if positivos == 2 and espacios == 2: s -= 20
            if negativos == 4: s += 5000
            if positivos == 4: s -= 5000
            return s
        centro = tablero[:, columnas//2]
        puntaje += np.sum(centro == -1) * 4
        for fila in range(filas):
            fila_arr = tablero[fila]
            for col2 in range(columnas-3):
                puntaje += sw(fila_arr[col2:col2+4])
        for col2 in range(columnas):
            col_arr = tablero[:, col2]
            for fila2 in range(filas-3):
                puntaje += sw(col_arr[fila2:fila2+4])
        for fila2 in range(filas-3):
            for col2 in range(columnas-3):
                diag1 = np.array([tablero[fila2+i, col2+i] for i in range(4)])
                diag2 = np.array([tablero[fila2+i, col2+3-i] for i in range(4)])
                puntaje += sw(diag1)
                puntaje += sw(diag2)
        self.tabla_valores[llave] = puntaje
        return puntaje

    def _ro(self, estado, jugador):
        puntaje = 0
        for _ in range(5):
            sim = ConnectState(estado.board.copy(), estado.player)
            for i in range(4):
                movs = sim.get_free_cols()
                if not movs: break
                accion = self.rng.choice(movs)
                try:
                    sim = sim.transition(accion)
                except:
                    break
                ganador = sim.get_winner()
                if ganador != 0:
                    puntaje += 150 if ganador == jugador else -150
                    break
        return puntaje / 5

    def _is_trap_move(self, estado, accion, jugador):
        try:
            nuevo_estado = estado.transition(accion)
        except:
            return False
        oponente = -jugador
        for col in nuevo_estado.get_free_cols():
            try:
                if nuevo_estado.transition(col).get_winner() == oponente:
                    return True
            except:
                pass
        return False

    def _adjust_iterations(self, estado, jugador):
        oponente = -jugador
        base_iters = max(self.iteraciones, self.iter_minima)
        for col in estado.get_free_cols():
            try:
                if estado.transition(col).get_winner() == oponente:
                    return max(base_iters, int(base_iters * 2))
            except:
                pass
        peligro = False
        for col in estado.get_free_cols():
            try:
                if self._ow(estado.transition(col), oponente):
                    peligro = True
                    break
            except:
                pass
        heuristica = self._h(estado)
        if peligro or abs(heuristica) < 30:
            return int(base_iters * 1.8)
        if abs(heuristica) > 200:
            return max(self.iter_minima, int(base_iters * 0.8))
        return max(self.iter_minima, int(base_iters * 0.5))

    def act(self, s):
        tablero = np.array(s, copy=True)
        negativos = np.sum(tablero == -1)
        positivos = np.sum(tablero == 1)
        jugador = -1 if negativos == positivos else (1 if positivos < negativos else -1)
        estado = ConnectState(tablero, jugador)
        movs = estado.get_free_cols()
        if len(movs) == 1:
            return movs[0]
        for col in movs:
            try:
                if estado.transition(col).get_winner() == jugador:
                    return col
            except:
                pass
        seguros = []
        for col in movs:
            if not self._is_trap_move(estado, col, jugador):
                seguros.append(col)
        if seguros:
            movs = seguros
        oponente = -jugador
        raiz = MCTSNode()
        for mov in movs:
            raiz.hijos[mov] = MCTSNode(raiz, mov)
        iteraciones = self._adjust_iterations(estado, jugador)
        for _ in range(iteraciones):
            nodo = raiz
            estado_actual = ConnectState(estado.board.copy(), estado.player)
            while True:
                if estado_actual.is_final() or not nodo.expandido:
                    break
                mejor_score = -1e18
                mejor_hijo = None
                mejor_accion = None
                for accion, hijo in nodo.hijos.items():
                    u = hijo.uct_score(nodo.visitas, self.c)
                    if u > mejor_score:
                        mejor_score = u
                        mejor_hijo = hijo
                        mejor_accion = accion
                if mejor_hijo is None:
                    break
                try:
                    siguiente = estado_actual.transition(mejor_accion)
                except:
                    break
                self._actualizar_adp(estado_actual, mejor_accion, siguiente)
                estado_actual = siguiente
                nodo = mejor_hijo
            if not estado_actual.is_final():
                acciones_validas = estado_actual.get_free_cols()
                no_exploradas = [a for a in acciones_validas if a not in nodo.hijos]
                if no_exploradas:
                    no_trampa = [x for x in no_exploradas if not self._is_trap_move(estado_actual, x, jugador)]
                    if no_trampa:
                        accion = self.rng.choice(no_trampa)
                    else:
                        accion = self.rng.choice(no_exploradas)
                    try:
                        nuevo_estado = estado_actual.transition(accion)
                    except:
                        nuevo_estado = estado_actual
                    self._actualizar_adp(estado_actual, accion, nuevo_estado)
                    nuevo_nodo = MCTSNode(nodo, accion, nuevo_estado)
                    nodo.hijos[accion] = nuevo_nodo
                    if self._ow(nuevo_estado, oponente):
                        nuevo_nodo.bono -= 40
                    centro_col = estado.COLS // 2
                    if nuevo_estado.player == -1 and accion == centro_col:
                        nuevo_nodo.bono += 3
                    nuevo_nodo.bono += self._h(nuevo_estado) * 0.002
                    nuevo_nodo.bono += self._ro(nuevo_estado, jugador) * 0.05
                    nuevo_nodo.bono += self._v_adp(nuevo_estado) * 0.01
                    if len(no_exploradas) == 1:
                        nodo.expandido = (len(acciones_validas) == 1)
                    nodo = nuevo_nodo
                    estado_actual = nuevo_estado
                else:
                    nodo.expandido = True
            recompensa = self._roll(estado_actual, jugador)
            retrocede = nodo
            while retrocede:
                retrocede.visitas += 1
                retrocede.valor += recompensa
                retrocede = retrocede.padre
        mejor_mov = None
        mas_visitas = -1
        for accion, hijo in raiz.hijos.items():
            if hijo.visitas > mas_visitas:
                mas_visitas = hijo.visitas
                mejor_mov = accion
        return int(mejor_mov)

    def _roll(self, estado, jugador_objetivo):
        sim = ConnectState(estado.board.copy(), estado.player)
        pasos = 0
        while not sim.is_final() and pasos < self.limite_rollout:
            movs = sim.get_free_cols()
            if not movs:
                break
            sim = sim.transition(self.rng.choice(movs))
            pasos += 1
        ganador = sim.get_winner()
        return 1 if ganador == jugador_objetivo else (0.5 if ganador == 0 else 0)
