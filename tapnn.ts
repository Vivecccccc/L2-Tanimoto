const sim = 0.95;
const cosThres = 2 * sim / (1 + sim);
const alpha = (1 / sim) * ((1 + (1 / sim)) + Math.sqrt((1 + (1 / 0.95)) ** 2 - 4));

function tapnn(infoMap: Map<string, number[]>) {
  const normMap: Map<string, number> = new Map([...infoMap].map(([k, v]) => {
    return [k, l2Norm(v)];
  }));
  const sortedMap: Map<string, number[]> = sortByNorm(infoMap, normMap);
  const normedMap: Map<string, number[]> = new Map([...sortedMap].map(([k, v]) => {
    return [k, normalizeVec(v, normMap.get(k)!)];
  }));
  const maxFeature: number[] = genMaxFeature(normedMap);
  let resMap: Map<string, string[]> = new Map();
  
  let prefMap: Map<string, number[]> = new Map();
  let indexMap: Map<number, Array<{sign: string, normedQJ: number, normedPrefQJ: number}>> = new Map(maxFeature.map((k, i) => {
    return [i, []];
  }));
  let stepStone: Map<number, number> = new Map(maxFeature.map((k, i) => {
    return [i, 0];
  }));
  let accumulator: Map<string, number> = new Map([...normMap].map(([k, v]) => {
    return [k, 0];
  }));
  
  for (const elem of sortedmMap) {
    const sign: string = elem[0];
    const vecQ: number[] = elem[1];
    const vecQNorm: number[] = normedMap.get(sign)!;
    
    let flagPref: boolean = false;
    let j: number = 0;
    while (j < vecQNorm.length && !flagPref) {
      const id = vecQNorm[j];
      if (id > 0) {
        let pScore = prefScore(vecQNorm, maxFeature, j);
        if (pScore >= cosThres) {
          flagPref = true;
          prefMap.set(sign, [j, pScore]);
          break;
        }
      }
      j += 1;
    }
    while (j < vecQNorm.length && flagPref) {
      const id = vecQNorm[j];
      if (id > 0) {
        let subList: Array<{sign: string, normedQJ: number, normedPrefQJ: number}> = indexMap.has(j) ? indexMap.get(j)! : [];
        const normedQJ = vecQNorm[j];
        const normedPrefQJ = l2Norm(vecQNorm.slice(0, j - 1));
        const param: {sign: string, normedQJ: number, normedPrefQJ: number} = {sign, normedQJ, normedPrefQJ};
        subList.push(param);
        indexMap.set(j, subList);
      }
      j += 1;
    }
  }
  for (let elem of sortedMap) {
    const sign: string = elem[0];
    const vecQ: number[] = elem[1];
    const p: number = prefMap.get(sign)![0];
    const vecQNorm: number[] = normedMap.get(sign)!;
    const lenThres: number = (1 / alpha) * normMap.get(sign)!;
    for (let j = vecQ.length - 1; j >= 0; j--) {
      for (let k = stepStone.get(j)!; k < indexMap.get(j)!.length; k++) {
        const ctx = indexMap.get(j)!;
        const ctxSign: string = ctx[k].sign;
        const pScore = prefScore(vecQNorm, maxFeature, j);
        if (normMap.get(ctxSign)! <= lenThres) {
          stepStone.set(j, stepStone.get(j)! + 1);
        }
        else if (normMap.get(ctxSign)! > normMap.get(sign)! || ctxSign === sign) {
          break;
        }
        else if (accumulator.get(ctxSign)! > 0 || (pScore >= cosThres && l2Norm(vecQ.slice(0, p)) >= eq8(sim, sign, normMap, pScore))) {
          const ctxA: number = accumulator.get(ctxSign)!;
          const delta: number = normedMap.get(sign)![j] * normedMap.get(ctxSign)![j];
          const factor: number = l2Norm(normedMap.get(sign)!.slice(0, j)) * l2Norm(normedMap.get(ctxSign)!.slice(0, j));
          if (ctxA + delta + factor < cosThres) {
            accumulator.set(ctxSign, 0);
          }
          accumulator.set(ctxSign, ctxA + delta);
        }
      }
    }
    let effAccumulator: Map<string, number> = new Map([...acumulator].filter(([k, v]) => v > 0));
    for (let ctx of effAccumulator) {
      const ctxSign: string = ctx[0];
      let ctxAcc: number = ctx[1];
      const ctxPref: number[] = prefMap.get(ctxSign)!;
      const ctxPPos: number = ctxPref[0];
      const ctxPScore: number = ctxPref[1];
      const ctxNormedVec: number[] = normedMap.get(ctxSign)!;
      let preEscape: boolean = false;
      
      if (ctxAcc + ctxPScore < cosThres) {
        effAccumulator.delete(ctxSign);
        accumulator.set(ctxSign, 0);
        continue;
      }
      let s: number = ctxAcc + ctxPScore;
      let beta: number = (s / cosThres) + Math.sqrt((s / cosThres) ** 2 - 1);
      if (normMap.get(ctxSign)! * beta < normMap.get(sign)!) {
        effAccumulator.delete(ctxSign);
        accumulator.set(ctxSign, 0);
        continue;
      }
    
      for (let j = Math.min(ctxPPos, vecQNorm.length) - 1; j >= 0; j--) {
        ctxAcc = effAccumulator.get(ctxSign)!;
        effAccumulator.set(ctxSign, ctxAcc + ctxNormedVec[j] * vecQNorm[j]);
        let delta: number = l2Norm(vecQNorm.slice(0, j)) * l2Norm(ctxNormedVec.slice(0, j));
        if (effAccumulator.get(ctxSign)! + delta < cosThres) {
          effAccumulator.delete(ctxSign);
          accumulator.set(ctxSign, 0);
          preEscape = true;
          break;
        }
      }
      if (preEscape) {
        continue;
      }
      const score = simScore(effAccumulator.get(ctxSign)!, normMap.get(sign)!, normMap.get(ctxSign)!);
      if (score > 0.95) {
        let subList: string[] = resMap.has(sign) ? resMap.get(sign)! : [];
        subList.push(ctxSign);
        resMap.set(sign, subList);
      }
    }
  }
}

funtion l2Norm(vec: number[]): number {
  let sum: number = 0;
  for (let i = 0; i < vec.length; i++) {
    const elem = vec[i];
    sum += elem ** 2;
  }
  return Math.sqrt(sum);
}

function sortByNorm(infoMap: Map<string, number[]>, normMap: Map<string, number>): Map<string, number[]> {
  let sortedMap: Map<string, number[]> = new Map([...infoMap].sort(a, b) => {
    return normMap.get(a[0])! - normMap.get(b[0])!;
  }));
  return sortedMap;
}

function normalizeVec(vec: number[], norm?: number): number[] {
  if (!norm) {
    norm = l2Norm(vec);
  }
  return vec.map(i => i / norm!);
}

function genMaxFeature(normedMap: Map<string, number[]>): number[] {
  let arr: number[][] = Array.from([...normedMap].map(([k, v]) => v));
  let maxFeature: number[] = arr.reduce(function(final, current) {
    let copiedFinal = Array.from(final);
    for (let i = 0; i < current.length; ++i) {
      if (!copiedFinal[i]) {
        copiedFinal[i] = 0;
      }
      if (current[i] > copiedFinal[i]) {
        copiedFinal[i] = current[i];
      }
    }
    return copiedFinal;
  });
  return maxFeature;
}

function innerProd(curVec: number[], refVec: number[]): number {
  let i: number = 0;
  let res: number = 0;
  while (i < Math.min(curVec.length, refVec.length)) {
    res += curVec[i] * refVec[i];
    i += 1;
  }
  return res;
}

function prefScore(curVec: number[], maxFeature: number[], pref: number): number {
  const prefVec: number[] = curVec.slice(0, pref + 1);
  let prefNorm: number = l2Norm(prefVec);
  let prefProd: number = innerProd(prefVec, maxFeature);
  return prefNorm < prefProd ? prefNorm : prefProd;
}

function eq8(eps: number, sign: string, normMap: Map<string, number>, pScore: number): number {
  const qNorm: number = normMap.get(sign)!;
  const iNorm: number = normMap.get(normMap.keys().next().value)!;
  const qm1: [string, number] = [...normMap].reduce(function (prev, curr) {
    if (curr[0] === sign || prev[0] === sign) {
      return prev;
    }
    return curr;
  });
  const factor_1: number = eps / (1 + eps);
  const factor_2: number = (qNorm ** 2 + iNorm ** 2) / (qm1[1] * pScore);
  return factor_1 * factor_2;
}

function simScore(acc: number, normQ: number, normCtx: number) {
  return acc / ((normQ ** 2 + normCtx ** 2) / (normQ * normCtx) - acc);
}
