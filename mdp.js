const EPS = 1e-12;

// Marsaglia–Tsang gamma (shape k > 0, scale theta=1)
function gammaRand(k, theta = 1) {
    if (k < 1) {
        // Johnk's / boosting trick: Gamma(k) = Gamma(k+1) * U^{1/k}
        const u = Math.random();
        return gammaRand(k + 1, theta) * Math.pow(u, 1 / k);
    }
    const d = k - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
        // standard normal
        let x, v;
        do {
            const u1 = Math.random(), u2 = Math.random();
            const R = Math.sqrt(-2 * Math.log(u1));
            const theta2 = 2 * Math.PI * u2;
            x = R * Math.cos(theta2); // N(0,1)
            v = 1 + c * x;
        } while (v <= 0);
        v = v * v * v;
        const u = Math.random();
        if (u < 1 - 0.0331 * (x * x) * (x * x)) return theta * d * v;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return theta * d * v;
    }
}

function dirichletRand(alphaVec) {
    const g = alphaVec.map(a => Math.max(EPS, gammaRand(a, 1)));
    return math.divide(g, math.sum(g));
    // return l1Normalize(g);
}

class MDP {
    /**
     * @param {number} numStates - number of states
     * @param {number} numActions - number of actions
     * @param {Array} transition - [numStates, numActions, numStates]
     * @param {Array} reward - [numStates, numActions]
     * @param {number} gamma - discount factor
     * @param {Array} p0 - [numStates]
     */
    constructor(numStates, numActions, transition, reward, gamma, p0) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.transition = transition;
        this.reward = reward;
        this.gamma = gamma;
        this.p0 = p0;

        // shape check
        const tShape = math.shape(this.transition);
        const rShape = math.shape(this.reward);
        const p0Shape = math.shape(this.p0);

        assert(
            tShape.length === 3 && tShape[0] === numStates && tShape[1] === numActions && tShape[2] === numStates,
            `transition must have shape [${numStates}, ${numActions}, ${numStates}], got ${JSON.stringify(tShape)}`
        );

        assert(
            rShape.length === 2 && rShape[0] === numStates && rShape[1] === numActions,
            `reward must have shape [${numStates}, ${numActions}], got ${JSON.stringify(rShape)}`
        );

        assert(
            p0Shape.length === 1 && p0Shape[0] === numStates,
            `p0 must have shape [${numStates}], got ${JSON.stringify(p0Shape)}`
        );
    }

    copy() {
        return new MDP(
            this.numStates,
            this.numActions,
            math.clone(this.transition),
            math.clone(this.reward),
            this.gamma,
            math.clone(this.p0)
        );
    }
}

const shuffleInPlace = (a) => {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
};


function generate_random_mdp(numStates, numActions, gamma, dirichletParam = 0.1) {
    const S = numStates, A = numActions;
    const T = math.zeros([S, A, S]);
    const K = Math.min(4, S);

    for (let s = 0; s < S; s++) {
        for (let a = 0; a < A; a++) {
            // pick K positions to be nonzero
            const idx = Array.from({ length: S }, (_, i) => i);
            shuffleInPlace(idx);
            const picked = idx.slice(0, K);

            // Dirichlet(K, alpha)
            const probsK = dirichletRand(Array(K).fill(dirichletParam));

            // place into length-S vector
            const row = Array(S).fill(0);
            for (let i = 0; i < K; i++) row[picked[i]] = probsK[i];

            // final numeric hygiene
            for (let sp = 0; sp < S; sp++) T[s][a][sp] = row[sp];

            // sanity
            let sumRow = 0;
            for (let sp = 0; sp < S; sp++) sumRow += T[s][a][sp];
            assert(Math.abs(sumRow - 1) < 1e-9, 'transition row not normalized');
        }
    }

    const R = math.zeros([S, A]);
    for (let s = 0; s < S; s++) {
        for (let a = 0; a < A; a++) {
            R[s][a] = Math.random();
        }
    }
    const p0 = math.zeros([S]);
    p0[0] = 1;
    return new MDP(S, A, T, R, gamma, p0);
}


// ------ core: policy evaluation ------
/**
 * @param {MDP} mdp
 * @param {Array<Array<number>>} pi - [S,A], row-stochastic
 * @returns {[Array<number>, Array<Array<number>>]} [v, q]
 */
function policy_evaluation_mdp(mdp, pi) {
    const S = mdp.numStates, A = mdp.numActions;
    const T = mdp.transition; // [S,A,S]
    const R = mdp.reward;     // [S,A]

    const r = math.sum(math.dotMultiply(R, pi), 1);

    // p[s][s'] = sum_a pi[s,a] * T[s,a,s']
    const p = math.zeros([S, S]);
    for (let s = 0; s < S; s++) {
        for (let a = 0; a < A; a++) {
            for (let s1 = 0; s1 < S; s1++) {
                p[s][s1] += pi[s][a] * T[s][a][s1];
            }
        }
    }

    // v = inv(I - gamma * p) * r
    const I = math.identity([S]);
    const G = math.subtract(I, math.multiply(mdp.gamma, p)); // (S,S)
    const invG = math.inv(G);
    const v = math.multiply(invG, r); // (S,)

    // q[s,a] = R[s,a] + gamma * sum_{s'} T[s,a,s'] * v[s']
    const q = math.zeros([S, A]);
    for (let s = 0; s < S; s++) {
        for (let a = 0; a < A; a++) {
            let adv = 0;
            for (let s1 = 0; s1 < S; s1++) adv += T[s][a][s1] * v[s1];
            q[s][a] = R[s][a] + mdp.gamma * adv;
        }
    }

    return [v, q];
}


/**
 * Compute discounted stationary distribution d^π(s,a) for γ<1.
 * Shapes:
 *  - T: [S][A][S], pi: [S][A], p0: [S]
 * Returns:
 *  - dPiSA: [S][A]
 */
function compute_stationary_distribution(mdp, pi) {
    const S = mdp.numStates, A = mdp.numActions, gamma = mdp.gamma;
    if (!(gamma < 1)) throw new Error('computeStationaryDistribution assumes gamma < 1');

    const T = mdp.transition;   // [S][A][S] (Array)
    const p0 = mdp.p0;          // [S]

    const SA = S * A;
    const idx = (s, a) => s * A + a;

    // b = (1-γ) * p0_pi, where p0_pi[(s,a)] = p0[s] * pi[s][a]
    const b = new Array(SA);
    for (let s = 0; s < S; s++) {
        for (let a = 0; a < A; a++) {
            b[idx(s, a)] = (1 - gamma) * p0[s] * pi[s][a];
        }
    }

    // Build P^π over state-actions: P[(s,a),(s',a')] = T[s][a][s'] * pi[s'][a']
    const P = math.zeros([SA, SA]); // Array [SA][SA]
    for (let s = 0; s < S; s++) {
        for (let a = 0; a < A; a++) {
            const row = idx(s, a);
            for (let sp = 0; sp < S; sp++) {
                for (let ap = 0; ap < A; ap++) {
                    const col = idx(sp, ap);
                    P[row][col] += T[s][a][sp] * pi[sp][ap];
                }
            }
        }
    }

    // Solve (I - γ P^T) d = b
    const PT = math.transpose(P);                          // [SA][SA]
    const I = math.identity([SA]);                // [SA][SA]
    const G = math.subtract(I, math.multiply(gamma, PT)); // [SA][SA]
    const d = math.multiply(math.inv(G), b); // [SA]
    console.log(d);

    // reshape back to [S][A]
    const dPiSA = math.zeros([S, A]);
    for (let s = 0; s < S; s++) {
        for (let a = 0; a < A; a++) {
            dPiSA[s][a] = d[idx(s, a)];
        }
    }
    return dPiSA;
}

/**
 * Compute discounted stationary **state** distribution d^π(s) for γ<1.
 * P^π_s[s][s'] = ∑_a π[s][a] T[s][a][s']
 * Returns:
 *  - dPiS: [S]
 */
function compute_state_stationary_distribution(mdp, pi) {
    const S = mdp.numStates, A = mdp.numActions, gamma = mdp.gamma;
    if (!(gamma < 1)) throw new Error('computeStateStationaryDistribution assumes gamma < 1');

    const T = mdp.transition; // [S][A][S]
    const p0 = mdp.p0;        // [S]

    // Build P_s^π: [S][S]
    const Ps = math.zeros(S, S).toArray();
    for (let s = 0; s < S; s++) {
        for (let a = 0; a < A; a++) {
            const w = pi[s][a];
            if (w === 0) continue;
            const Tsas = T[s][a]; // [S']
            for (let sp = 0; sp < S; sp++) {
                Ps[s][sp] += w * Tsas[sp];
            }
        }
    }

    // Solve (I - γ P_s^T) d = (1-γ) p0
    const PsT = math.transpose(Ps);
    const I = math.identity(S).toArray();
    const G = math.subtract(I, math.multiply(gamma, PsT));
    const rhs = math.multiply(1 - gamma, p0);
    const d = math.multiply(math.inv(G), rhs); // [S]

    return d;
}


/**
 * Solve MDP via policy iteration.
 * @param {MDP} mdp - An MDP instance.
 * @returns {[Array<Array<number>>, Array<number>, Array<Array<number>>]} [pi, v, q]
 */
function solve_mdp_dp(mdp) {
    const S = mdp.numStates, A = mdp.numActions;

    // 초기 정책: uniform
    let pi = math.divide(math.ones([S, A]), A);
    let vOld = math.zeros([S]);

    let v, q, pi_new;
    for (let iter = 0; iter < 1_000_000; iter++) {
        [v, q] = policy_evaluation_mdp(mdp, pi);

        // Greedy policy update
        pi_new = math.zeros(S, A).toArray();
        for (let s = 0; s < S; s++) {
            // argmax q[s]
            let maxVal = -Infinity, maxIdx = 0;
            for (let a = 0; a < A; a++) {
                if (q[s][a] > maxVal) {
                    maxVal = q[s][a];
                    maxIdx = a;
                }
            }
            pi_new[s][maxIdx] = 1;
        }

        // convergence check
        const samePolicy = pi.every((row, s) => row.every((val, a) => val === pi_new[s][a]));
        const vDiff = math.max(math.abs(math.subtract(v, vOld)));

        if (samePolicy || vDiff < 1e-8) {
            pi = pi_new;
            break;
        }

        vOld = v;
        pi = pi_new;
    }

    return [pi, v, q];
}
