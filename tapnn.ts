import { DisplayDupViewProvider } from "./DisplayDupViewProvider";

const cosThres = 2 * 0.95 / (1 + 0.95);
const alpha = (1 / 2) * ((1 + (1 / 0.95)) + Math.sqrt((1 + (1 / 0.95)) ** 2 - 4));

export async function tapnn(infoMap: Map<string, number[]>, wvvProvider: DisplayDupViewProvider) {
    const rawNormMap: Map<string, number> = new Map([...infoMap].map(([k, v]) => {
        return [k, l2Norm(v)];
    }));
    const normMap: Map<string, number> = sortNorm(rawNormMap);
    const normArr: [string, number][] = [...normMap];
    const sortedMap: Map<string, number[]> = sortByNorm(infoMap, normMap);
    const sortedArr: [string, number[]][] = [...sortedMap];
    const normedMap: Map<string, number[]> = new Map([...sortedMap].map(([k, v]) => {
        return [k, normalizeVec(v, normMap.get(k)!)];
    }));
    const normedArr: [string, number[]][] = [...normedMap];

    const cumNormMap: Map<string, number[]> = new Map(sortedArr.map(([k, v]) => {
        return [k, cumSqSum(v)];
    }));
    const cumNormArr: [string, number[]][] = [...cumNormMap];

    const maxFeature: number[] = genMaxFeature(normedMap);
    let visitedMap: Map<string, boolean> = new Map(sortedArr.map(([k, v]) => {
        return [k, false];
    }));
    let resMap: Map<string, string[]> = new Map();

    let prefMap: Map<string, number[]> = new Map();
    let indexMap: Map<number, Array<{sign: string, normedQJ: number, normedPrefQJ: number}>> = new Map(maxFeature.map((k, i) => {
        return [i, []];
    }));
    let stepStone: Map<number, number> = new Map(maxFeature.map((k, i) => {
        return [i, 0];
    }));
    let accumulator: Map<string, number> = new Map(normArr.map(([k, v]) => {
        return [k, 0];
    }));
    for (let i = 0; i < sortedArr.length; i++) {
        const elem: [string, number[]] = sortedArr[i];
        const sign: string = elem[0];
        const vecQ: number[] = elem[1];
        const vecQNorm: number[] = normedMap.get(sign)!;

        let flagPref: boolean = false;
        let j: number = 0;
        while (j < vecQNorm.length && !flagPref) {
            const id = vecQNorm[j];
            if (id > 0) {
                let pScore = prefScore(sign, maxFeature, cumNormMap, normedMap, j);
                if (pScore >= cosThres) {
                    flagPref = true;
                    prefMap.set(sign, [j, prefScore(sign, maxFeature, cumNormMap, normedMap, j - 1)]);
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
                const normedPrefQJ = Math.sqrt(cumNormMap.get(sign)![j - 1]);
                const param: {sign: string, normedQJ: number, normedPrefQJ: number} = {sign, normedQJ, normedPrefQJ};
                subList.push(param);
                indexMap.set(j, subList);
            }
            j += 1;
        }
    }
    for (let i = 0; i < sortedArr.length; i++) {
        const elem: [string, number[]] = sortedArr[i];
        const sign: string = elem[0];
        const vecQ: number[] = elem[1];
        const p: number = prefMap.get(sign)![0];
        const vecQNorm: number[] = normedMap.get(sign)!;
        const lenThres: number = (1 / alpha) * normMap.get(sign)!;
        accumulator = new Map(normArr.map(([k, v]) => {
            return [k, 0];
        }));
        for (let j = vecQ.length - 1; j >= 0; j--) {
            const pScore = prefScore(sign, maxFeature, cumNormMap, normedMap, j);
            for (let k = stepStone.get(j)!; k < indexMap.get(j)!.length; k++) {
                const ctx = indexMap.get(j)!;
                const ctxSign: string = ctx[k].sign;
                if (visitedMap.get(ctxSign)) {
                    accumulator.set(ctxSign, 0);
                    continue;
                }
                if (normMap.get(ctxSign)! <= lenThres) {
                    stepStone.set(j, stepStone.get(j)! + 1);
                }
                else if (normMap.get(ctxSign)! > normMap.get(sign)! || ctxSign === sign) {
                    break;
                }
                else if (accumulator.get(ctxSign)! > 0 || (pScore >= cosThres && Math.sqrt(cumNormMap.get(sign)![p - 1]) >= eq8_2(0.95, i, normArr, pScore))) {    
                    const ctxA: number = accumulator.get(ctxSign)!;
                    const delta: number = normedMap.get(sign)![j] * normedMap.get(ctxSign)![j];
                    const factor: number = Math.sqrt((cumNormMap.get(sign)![j - 1]) * (cumNormMap.get(ctxSign)![j - 1]));
                    accumulator.set(ctxSign, ctxA + delta);
                    if (ctxA + delta + factor < cosThres) {
                        accumulator.set(ctxSign, 0);
                    }
                }
            }
        }
        let effAccumulator: Map<string, number> = new Map([...accumulator].filter(([k, v]) => v > 0));
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
                const delta: number = Math.sqrt((cumNormMap.get(sign)![j - 1]) * (cumNormMap.get(ctxSign)![j - 1]));
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
                let subList: string[] = resMap.has(ctxSign) ? resMap.get(ctxSign)! : [];
                subList.push(sign);
                resMap.set(ctxSign, subList);
                visitedMap.set(sign, true);
            }
        }
        await wvvProvider.view?.webview.postMessage({command: 'refprog', result: `${100 * (i + 1) / sortedArr.length}`});
    }
    return resMap;
}

function l2Norm(vec: number[]): number {
    let sum: number = 0;
    for (let i = 0; i < vec.length; i++) {
        const elem = vec[i];
        sum += elem ** 2;
    }
    return Math.sqrt(sum);
}

function sortByNorm(infoMap: Map<string, number[]>, normMap: Map<string, number>): Map<string, number[]> {
    let sortedMap: Map<string, number[]> = new Map([...infoMap].sort((a, b) => {
        return normMap.get(a[0])! - normMap.get(b[0])!;
    }));
    return sortedMap;
}

function sortNorm(normMap: Map<string, number>): Map<string, number> {
    let sortedMap: Map<string, number> = new Map([...normMap].sort((a, b) => {
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

function prefScore(sign: string, maxFeature: number[], cumNormMap: Map<string, number[]>, normedMap: Map<string, number[]>, pref: number): number {
    const prefVec: number[] = normedMap.get(sign)!.slice(0, pref + 1);
    const prefNorm: number = Math.sqrt(cumNormMap.get(sign)![pref]);
    const prefProd: number = innerProd(prefVec, maxFeature);
    return prefNorm < prefProd ? prefNorm : prefProd
}

function eq8(eps: number, sign: string, normMap: Map<string, number>, pScore: number): number {
    const qNorm: number = normMap.get(sign)!;
    const iNorm: number = [...normMap][0][1];
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

function eq8_2(eps: number, i: number, normArr: [string, number][], pScore: number): number {
    const qNorm: number = normArr[i][1];
    const iNorm: number = normArr[0][1];
    const qm1: number = normArr[i - 1][1];
    const factor_1: number = eps / (1 + eps);
    const factor_2: number = (qNorm ** 2 + iNorm ** 2) / (qm1 * pScore);
    return factor_1 * factor_2;
}

function simScore(acc: number, normQ: number, normCtx: number) {
    return acc / ((normQ ** 2 + normCtx ** 2) / (normQ * normCtx) - acc);
}

function cumSqSum(arr: number[]) {
    return arr.map((sum => (val: number) => sum += val ** 2)(0));
}
