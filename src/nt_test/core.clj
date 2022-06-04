(ns nt-test.core
  (:require
   [nt-test.sim :as sim :refer 
    [pmap-vals logi logi-adj dist dist-sq mv tr ones p-fn disj-vec genkey
     rand-double rand-gaussian rand-uniform add-gaussian add-uniform randbetween]])
  (:gen-class))

;; no main, runs in repl
(defn -main
  [& args]
  ())

;; default functions
(def default-act-fn logi-adj)
(def mu-wt-fn add-gaussian)
(def set-wt-fn rand-uniform)

;; a gene representing a connection between nodes in the neural network
(defrecord cnxGene [in out wt on?])
;; a genome defines a network; consists of nodes and connections
;;   nodes is an integer-keyed map of keywords identifying node types
;;   connections is an integer-keyed map of cnxGenes
;;   in-nodes identifies environmental inputs and bias nodes
(defrecord fnGenome [nodes connections in-nodes])

;; measures toplogical distance between genomes
(defn diff-cnx [{m1 :connections} {m2 :connections}]
  (let [s1 (set (keys m1))
        s2 (set (keys m2))
        n (max (count s1) (count s2))
        n1 (count (filter (complement s2) s1))
        n2 (count (filter (complement s1) s2))]
    (/ (+ n1 n2) n) 
    ))

;; measures difference in connection weights
(defn diff-wt [{m1 :connections} {m2 :connections}]
  (let [ks (keys m2)]
    ;;mean squared distance
    (/ (dist-sq
        (keep #(get-in m1 [% :wt]) ks)
        (keep #(get-in m2 [% :wt]) ks)) 
       (count ks))
    ))

;; Node protocol currently only has one type, fNode
(defprotocol Node
  (add-input [x in wt])
  (add-output [x out])
  (remove-input [x k])
  (remove-output [x k])
  (node-inputs [x])
  (node-wts [x])
  (node-outputs [x])
  (node-f [x]))

;; a neural network node with activation function f 
;;   in-map and out-map contain the data about network connections and weights
(defrecord fNode [in-map out-map f]
  Node
  (add-input [x in wt] (assoc-in x [:in-map in] wt))
  (add-output [x out] (assoc-in x [:out-map out] 1))
  (remove-input [x k] (update x :in-map dissoc k))
  (remove-output [x k] (update x :out-map dissoc k))
  (node-inputs [x] (keys in-map))
  (node-wts [x] (vals in-map))
  (node-outputs [x] (keys out-map))
  (node-f [x] f))

;; Netw protocol currently has only one type, fNet
(defprotocol Netw
  (add-cnx [x c])
  (add-update-cnxs [x connections])
  (eval-net [x prev-vals out-nodes]))

;; an evaluate-able neural network
;;  netw is an integer-keyed map of fNodes
(defrecord fNet [netw in-nodes]
  Netw

  ;;add-cnx takes a cnxGene and adds network connections to the appropriate nodes
  (add-cnx [x {:keys [in out wt on?]}]
    (if on?
      (-> x
          (update-in [:netw in] add-output out)
          (update-in [:netw out] add-input in wt))
      (-> x
          (update-in [:netw in] remove-output out)
          (update-in [:netw out] remove-input in))
      ))

  ;;takes a collection of cnxGenes and calls add-cnx on all of them
  (add-update-cnxs [x connections]
                   (reduce add-cnx x connections))

  ;;calculates and returns output values for all nodes in the network 
  ;;  (traverses backwards from output nodes)
  (eval-net [{:keys [netw]} prev-vals out-nodes]
    (loop [todo (into [] out-nodes)
           active #{}
           val-map prev-vals]
      (if (empty? todo)
        val-map
        (let [curr-idx (peek todo)
              act-idxs (conj active curr-idx)
              curr-node (netw curr-idx)
              child-idxs (node-inputs curr-node)
              inact-children (apply disj (set child-idxs) act-idxs)
              ready? (empty? inact-children)]
          #_(if ready?
              (println curr-idx "reading from" child-idxs)
              (println curr-idx "not ready"))
          (recur
           (if ready?
             (into (disj-vec todo curr-idx) inact-children)
             (into todo inact-children))
           act-idxs
            ;; if node has no inputs... leave init val unchanged
           (if (and ready? (seq child-idxs))
             (assoc val-map curr-idx
                    (mapv (node-f curr-node)
                          (mv (mapv val-map child-idxs) (node-wts curr-node))))
             val-map))
          
          ))))
  )

;; constructs a network data structure and adds nodes
;; doesn't have connections yet -- these have to be added at the node level
(defn fnet-template [node-type-map]
  (let [n (reduce (fn [m [k v]]
                    (assoc m k (if (#{:in :bias} v)
                                 (fNode. {k 1.0} {} identity)
                                 (fNode. {} {} default-act-fn))))
                  {}
                  node-type-map)
        i (into [] (keys (filter #(#{:in :bias} (val %)) node-type-map)))]
    (fNet. n i)))

;; generates an evalate-able network data structure from a genome
(defn gen-netw [{:keys [nodes connections]}]
  (add-update-cnxs (fnet-template nodes) (vals connections)))

;; mutates connection weight using global default mutation function
(defn mu-wt [cx]
  (update cx :wt mu-wt-fn))
;; same as mu-wt but doesn't care about the current value
(defn replace-wt [cx]
  (assoc cx :wt (set-wt-fn)))

;; global integer for indexing new genes
(def g-idx (atom 0))

;; adds a new connection to a genome and updates the resulting network
;;   returns a map containing both
(defn mu-add-cnx [{:keys [genome net]}]
  (let [{:keys [nodes connections in-nodes]} genome
        node-ids (keys nodes)
        from-node (rand-nth node-ids)
        ;;
        ;; ? disallow self-connections ?
        ;;disallowed-to (conj in-nodes from-node)
        ;;allowed-to (filterv (complement disallowed-to) node-ids)
        allowed-to (filterv (complement in-nodes) node-ids)
        to-node (rand-nth allowed-to)]

    (if-let
     [[i c] (first (filter
                    (fn [[k {:keys [in out]}]] (and (= from-node in) (= to-node out)))
                    connections))]
      ;cnxn already exists
      ;;don't want to waste an eval... maybe mutate weight?
      (let [upd-c
            ;; ? if it's off, turn it back on ?
            (if (:on? c)
              (mu-wt c)
              (assoc c :on? true))
            ]
        {:genome (assoc-in genome [:connections i] upd-c)
         :net (add-cnx net upd-c)})
      ;;add new
      (let [c (cnxGene. from-node to-node (set-wt-fn) true)]
        {:genome (assoc-in genome [:connections (swap! g-idx inc)] c)
        :net (add-cnx net c)}
        ))))

;; adds a new node to a genome and updates the resulting network
;;   returns a map containing both
(defn mu-add-node [{:keys [genome net]}]
  (let [{:keys [nodes connections]} genome
        ;;adding a node
        new-idx (inc (apply max (keys nodes)))
      ;;break a random connection
        ;;
        ;; TODO -- allow active connections only??
        ;; 
        cnx-idx (rand-nth (keys connections))
        {orig-in :in orig-out :out wt :wt on? :on?} (get connections cnx-idx)
        ;;if the connection is disabled...
        ;;  add the node anyway but disable the new connections
        c1 (cnxGene. orig-in new-idx 1.0 on?)
        c2 (cnxGene. new-idx orig-out wt on?)]

    {:genome (-> genome
                 (update :nodes assoc new-idx :h)
                 (update :connections assoc-in [cnx-idx :on?] false)
                 (update :connections assoc 
                         (swap! g-idx inc) c1 
                         (swap! g-idx inc) c2))

     :net (-> net
              (update-in [:netw orig-in :out-map] dissoc orig-out)
              (update-in [:netw orig-out :in-map] dissoc orig-in)
              (assoc-in [:netw new-idx] (fNode. {} {} default-act-fn))
              (add-cnx c1)
              (add-cnx c2))}
    ))

;; helper function to mutate connection weights
(defn jitter-weights [g]
  (assoc g :connections
         (pmap-vals (p-fn 0.9 mu-wt replace-wt) (:connections g))))

;; mutates all connection weights in a genome and updates the resulting network
;;   returns a map containing both
(defn mu-cnx-weights [{:keys [genome net]}]
  (let [g-wts (jitter-weights genome)]
    {:genome g-wts
     :net (add-update-cnxs net (vals (:connections g-wts)))}
    ))

;; probabalistic mutation functions
;;    the numbers here determine mutation rates
(def p-add-node (p-fn 0.01 mu-add-node identity))
(def p-add-cnx (p-fn 0.1 mu-add-cnx identity))
(def p-cnx-wts (p-fn 0.8 mu-cnx-weights identity))

;; generates a mutant offspring
;;   m is a map containing keys :genome and :net for genome and resulting network
(defn mutant-clone [m]
  (-> m
      (p-cnx-wts)
      (p-add-node)
      (p-add-cnx)))

;;sum square distance between matching vectors in given maps
(defn dist-vecmap [val-m targ-m]
  (apply + (map (fn [k] (dist-sq (val-m k) (targ-m k))) (keys targ-m))))

;;recurrent connections are for sequential inputs only
;;  -- init vals should be reset each generation
;;make sure new nodes get init vals
(defn eval-fresh [nn input-map out-nodes]
  (eval-net nn
             (into
              (zipmap (keys (:netw nn)) 
                      (repeat (ones (val (first input-map)))))
              input-map)
            out-nodes))

;; evaluates network outputs and calculates cost function using sum sq dist to targ-map
(defn cost-target-output [net in-map targ-map]
  (let [v (eval-fresh net in-map (keys targ-map))]
    [(dist-vecmap v targ-map) v]))


;; clone-and-eval generates and evaluates a mutant offspring
;; takes a parent and a cost function
;; parent is a map containing the following keys
;;   :genome is an fnGenome
;;   :net is a Netw
;;   :vals is a map containing output values of network nodes
;;   :cost is the result of the cost function
;;   :dish-no is dish assignment (dishes are separate populations kept to maintain diversity)
;;   :dist is distance from parent
;;   :age-cost and :st-cost for age cost and statsis cost (see explanation below)
;;  TODO: define a record for individuals
(defn clone-and-eval 
  [{p-gen :genome p-dist :dist p-dn :dish-no 
    p-cost :cost p-agec :age-cost p-st :st-cost 
    :as p} 
   cost-fn]
  (let [{:keys [net genome] :as o} (mutant-clone p)
        [c v] (cost-fn net)
        ;;measures improvement over time
        ;; -- remains unchanged when less-fit offspring founds a new dish
        delta-c (- c p-cost)
        topo-dist-founder (+ p-dist (diff-cnx genome p-gen))
        wt-dist-parent (diff-wt genome p-gen)
        ;;when genetic distance from founder exceeds threshold,
        ;;individual becomes founder of a new dish & doesn't compete with parent
        ;;currently using topological novelty as distance, and preserving any novelty
        ;;founding a new dish decreases age-cost (tracks which dishes are newest)
        ;;when offspring replaces parent, stasis cost decreases (detects lineages stuck at optima)
        ;; -- new dish inherits parent's progress
        [o-dist o-dn o-agec o-st] (if (or
                                  (> topo-dist-founder 0)
                                  ;;founding a new dish on large weight change 
                                  ;;  should help escape local optima
                                  (> wt-dist-parent 3.0))
                                      [0 (genkey :n) (dec p-agec) p-st]
                                      [p-dist p-dn p-agec (+ p-st delta-c)])]
    (assoc o :vals v :cost c :dish-no o-dn 
           :dist o-dist :age-cost o-agec :st-cost o-st
           )))



;; experiments


;; xor experiment
(let [nds {1 :in, 2 :in, 3 :bias, 4 :out, 5 :h}
      cnxs {1 (cnxGene. 1 4 (set-wt-fn) true)
            2 (cnxGene. 2 4 (set-wt-fn) true)
            3 (cnxGene. 3 4 (set-wt-fn) true)}
      in-nodes #{1 2 3}
      g (fnGenome. nds cnxs in-nodes)
      nn (gen-netw g)

      in-map {1 [1 1 0 0] 2 [1 0 1 0]}
      targ {4 [0 1 1 0]}

      [c out-vals] (cost-target-output nn in-map targ)
      ;;track  :dish-no (dish number)
      ;;:dist (genetic distance from dish founder)
      ;;:age-cost (relative genetic novelty) :st-cost (fitness improvement)
      i {:genome g :net nn :vals out-vals :cost c
         :dish-no (genkey :n) :dist 0 :age-cost 0 :st-cost 0}]

  (reset! g-idx (apply max (keys cnxs)))
  (reset! curr-best [i]))

(let [in-map {1 [1 1 0 0] 2 [1 0 1 0]}
      targ {4 [0 1 1 0]}
      cost-fn #(cost-target-output % in-map targ)]
  (loop [x 2
         pvec @curr-best]
    (if (< x 1)
      (do (reset! curr-best pvec)
        ;nil
          (first (sort-by :cost @curr-best)))
      (let [pop-fn (fn [p] (into [p] (repeatedly 1 #(clone-and-eval p cost-fn))))
            pop (apply concat (pmap pop-fn pvec))
            dish-champs (pmap-vals #(apply min-key :cost %) (group-by :dish-no pop))

            most-fit (take 20 (keys (sort-by (comp :cost val) dish-champs)))
            best-new (take 10 (keys (sort-by (comp (juxt :age-cost :cost) val) dish-champs)))
            freshest (take 20 (keys (sort-by (comp :st-cost val) dish-champs)))
            fresh-new (take 0 (keys (sort-by (comp (juxt :age-cost :st-cost) val) dish-champs)))]
        (recur (dec x)
               (vals (select-keys dish-champs (concat most-fit best-new freshest fresh-new))))))))

;@curr-best




;; cart control experiment
(let [g 9.8
      mcart 1.0
      mpole 0.1
      mtot-inv (/ 1.0 (+ mcart mpole))
      lpole 0.5
      mxlpole (* mpole lpole)
      magf 10.0
      dt 0.02
      fourthirds (/ 4 3)]
  (defn push-cart [dir x x-dot th th-dot]
    (let [f (* dir magf)
          cos-th (Math/cos th)
          sin-th (Math/sin th)
          tmp (* mtot-inv (+ f (* mxlpole th-dot th-dot sin-th)))
          th-acc-n (- (* g sin-th) (* tmp cos-th))
          th-acc-d (* lpole (- fourthirds (* mpole cos-th cos-th mtot-inv)))
          th-acc (/ th-acc-n th-acc-d)
          x-acc (- tmp (* mxlpole th-acc cos-th mtot-inv))]
      (mapv #(+ %1 (* dt %2)) [x x-dot th th-dot] [x-dot x-acc th-dot th-acc])
      )))


(let [max-steps 100 ;100000
      twelve-deg (* 12.0 (/ Math/PI 180))
      neg-12-deg (- twelve-deg)
      x (- (/ (randbetween 0 4799) 1000.0) 2.4)
      x-dot (- (/ (randbetween 0 1999) 1000.0) 1.0)
      th (- (/ (randbetween 0 399) 1000.0) 0.2)
      th-dot (- (/ (randbetween 0 2999) 1000.0) 1.5)
      init-cart-state [x x-dot th th-dot]
      calc-inputs (fn [v] (zipmap [1 2 3 4]
                                  (mapv #(vector (/ (+ %1 %2) %3)) v
                                        [2.4 0.75 twelve-deg 1.0]
                                        [4.8 1.5 0.41 2.0])))
      init-inputs (calc-inputs init-cart-state)
      outA 6
      outB 7]
  (defn cost-cart-control [net]
    (loop [i max-steps
           [x x-dot th th-dot] init-cart-state
           node-vals (eval-fresh net init-inputs [outA outB])]
      (if (or (> th twelve-deg) (< th neg-12-deg) (> x 2.4) (< x -2.4) (< i 1))
        [(/ i max-steps) node-vals]
        (let [dir (if (> (first (node-vals outA)) (first (node-vals outB))) -1 1)
              cart-state (push-cart dir x x-dot th th-dot)
              new-vals (into node-vals (calc-inputs cart-state))]
          (recur (dec i)
                 cart-state
                 (eval-net net new-vals [outA outB])
                 )))
      )))


(def curr-best (atom {}))

(let [nds {1 :in, 2 :in, 3 :in, 4 :in, 5 :bias 6 :out 7 :out 8 :h 9 :h}
      cnxs {1 (cnxGene. 1 8 (set-wt-fn) true),   2 (cnxGene. 1 9 (set-wt-fn) true)
            3 (cnxGene. 2 8 (set-wt-fn) true),   4 (cnxGene. 2 9 (set-wt-fn) true)
            5 (cnxGene. 3 8 (set-wt-fn) true),   6 (cnxGene. 3 9 (set-wt-fn) true)
            7 (cnxGene. 4 8 (set-wt-fn) true),   8 (cnxGene. 4 9 (set-wt-fn) true)
            9 (cnxGene. 5 8 (set-wt-fn) true),  10 (cnxGene. 5 9 (set-wt-fn) true)
            11 (cnxGene. 8 6 (set-wt-fn) true), 12 (cnxGene. 8 7 (set-wt-fn) true)
            13 (cnxGene. 9 6 (set-wt-fn) true), 14 (cnxGene. 9 7 (set-wt-fn) true)
            }
      in-nodes #{1 2 3 4 5}
      g (fnGenome. nds cnxs in-nodes)
      nn (gen-netw g)

      [c out-vals] (cost-cart-control nn)
      ;;track  :dish-no (dish number)
      ;;:dist (genetic distance from dish founder)
      ;;:age-cost (relative genetic novelty) :st-cost (fitness improvement)
      i {:genome g :net nn :vals out-vals :cost c
         :dish-no (genkey :n) :dist 0 :age-cost 0 :st-cost 0}]

  (reset! g-idx (apply max (keys cnxs)))
  (reset! curr-best [i]))


(let [cost-fn cost-cart-control]
  (loop [x 100
         pvec @curr-best]
    (if (< x 1)
      (do (reset! curr-best pvec)
        ;nil
          (first (sort-by :cost @curr-best)))
      (let [pop-fn (fn [p] (into [p] (repeatedly 4 #(clone-and-eval p cost-fn))))
            pop (apply concat (pmap pop-fn pvec))
            dish-champs (pmap-vals #(apply min-key :cost %) (group-by :dish-no pop))

            most-fit (take 4 (keys (sort-by (comp :cost val) dish-champs)))
            best-new (take 2 (keys (sort-by (comp (juxt :age-cost :cost) val) dish-champs)))
            freshest (take 4 (keys (sort-by (comp :st-cost val) dish-champs)))
            fresh-new (take 0 (keys (sort-by (comp (juxt :age-cost :st-cost) val) dish-champs)))]
        (recur (dec x)
               (vals (select-keys dish-champs (concat most-fit best-new freshest fresh-new)))))))
  )

;(use 'nt-test.core :reload-all)




