import time

class QuadTree:
    id_counter = 0  # Static counter to assign unique IDs to leaves
    global_id_counter = 0

    def __init__(self, x_min, x_max, y_min, y_max, leaf_size, level=0,parent=None):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.leaf_size = leaf_size
        self.level = level  # Current depth level
        self.children = []
        self.parent = parent
        self.global_id = QuadTree.global_id_counter
        self.is_leaf = True
        self.id = QuadTree.id_counter  # Unique ID for leaves, assigned when it's a leaf
        self.repo_count = 0  # Initialize repo_count to 0
        QuadTree.global_id_counter += 1
        self.merged = 0

    def subdivide(self):
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2
        self.children = [
            QuadTree(self.x_min, x_mid, self.y_min, y_mid, self.leaf_size, self.level + 1,self),  # Bottom-left
            QuadTree(x_mid, self.x_max, self.y_min, y_mid, self.leaf_size, self.level + 1,self),  # Bottom-right
            QuadTree(self.x_min, x_mid, y_mid, self.y_max, self.leaf_size, self.level + 1,self),  # Top-left
            QuadTree(x_mid, self.x_max, y_mid, self.y_max, self.leaf_size, self.level + 1,self),  # Top-right
        ]
        self.is_leaf = False

    def build(self):
        if (self.x_max - self.x_min > self.leaf_size) or (self.y_max - self.y_min > self.leaf_size):
            self.subdivide()
            for child in self.children:
                child.build()
        # else:
        #     # Initially, no ID is assigned until reset_ids is called
        #     self.id = None

    # ids on leaf only
    def reset_ids_leafs(self):
        """Reset the IDs of all leaves in the quad tree."""
        QuadTree.id_counter = 0
        for leaf in self.get_leaves():
            leaf.id = QuadTree.id_counter
            QuadTree.id_counter += 1

    # def find_leaf(self, x, y):
    #     """Find the leaf node containing the given point (x, y), following only one branch."""
    #     if self.is_leaf:
    #         return self  # If it's already a leaf, return it

    #     # Determine which quadrant the point belongs to and recurse into that child
    #     x_mid = (self.x_min + self.x_max) / 2
    #     y_mid = (self.y_min + self.y_max) / 2

    #     if x < x_mid:
    #         if y < y_mid:
    #             return self.children[0].find_leaf(x, y)  # Bottom-left
    #         else:
    #             return self.children[2].find_leaf(x, y)  # Top-left
    #     else:
    #         if y < y_mid:
    #             return self.children[1].find_leaf(x, y)  # Bottom-right
    #         else:
    #             return self.children[3].find_leaf(x, y)  # Top-right

    def find_leaf(self, x, y):
        """Find the leaf node containing the given point (x, y), correctly checking non-uniform partitions."""
        if self.is_leaf:
            return self  # If it's already a leaf, return it

        # Iterate over children and find the one that contains (x, y)
        for child in self.children:
            if child.x_min <= x < child.x_max and child.y_min <= y < child.y_max:
                return child.find_leaf(x, y)
            
        return None  # Should never happen if (x, y) is within the tree bounds

    def reset_ids(self):
        """Reset the IDs of all nodes (both internal nodes and leaves) in the quad tree."""
        QuadTree.id_counter = 0

        self.traverse_and_reset()


    def traverse_and_reset(self):
        """Recursively traverse all nodes and reset their IDs."""
        self.id = QuadTree.id_counter
        QuadTree.id_counter += 1
        if self.children:  # If the node has children, recurse into them
            for child in self.children:
                child.traverse_and_reset()    


    def get_cell_count(self):
        """Return the number of cells in the quad tree."""
        if self.is_leaf:
            return 1
        return sum(child.get_cell_count() for child in self.children)

    def merge_children_drop_child(self, tr):
        """
        Merge a node's children into a single leaf if the sum of their counts
        is below the threshold `tr`.
        """
        if self.is_leaf:
            return

        # Check if the sum of children's repo_count is below the threshold
        child_repo_count = sum(child.repo_count for child in self.children)
        if child_repo_count < tr:
            # print("merge")
            self.children = []
            self.is_leaf = True
            self.repo_count = child_repo_count
            self.id = None  # Reset ID until reset_ids is called
        else:
            # Recursively apply to children
            for child in self.children:
                child.merge_children_drop_child(tr)

    def merge_children_drop_child_v2(self, tr):
        """
        Merge a node's children into a single leaf if the sum of their counts
        is below the threshold `tr`.
        """
        # print("id:",self.id)

        if not self.is_leaf:
            # print("Not Leaf")
            if self.repo_count < tr:
                # print("menor")    
                self.children = []
                self.is_leaf = True
                self.id = None  # Reset ID until reset_ids is called
            else:
                # Recursively apply to children
                for child in self.children:
                    child.merge_children_drop_child_v2(tr)    
        else:
            # print("Leaf")
            if self.repo_count < tr:
                print("merge")
                self.parent.children = []
                self.parent.is_leaf = True
                self.parent.id = None  # Reset ID until reset_ids is called
            else:
                return

    # Tem que pensar. Tenho que sair mergeando sem quebrar a estrutura.
    def merge_children(self, tr, mv):
        """
        Merge a node's children into a single leaf if the sum of their counts
        is below the threshold `tr`.
        """
        # print("new_merge")

        if self.parent is None and self.is_leaf:
            # print(self.id,self.repo_count)
            return
        
        else:

            if not self.is_leaf :
                if self.repo_count < tr:
                    self.parent.children = []
                    self.parent.is_leaf = True
                    self.id = None  # Reset ID until reset_ids is called
                    self.repo_count - self.parent.repo_count
                else:
                    for child in self.children:
                        
                        





                    # print("child:",child.id)
                        child.merge_children(tr,mv)

            else:
                #uma folha
                while self.repo_count < tr:
                    # print("self.repo_count:",self.repo_count)
                    # Get siblings from the parent
                    siblings = [child for child in self.parent.children if child is not self]
                    
                    # so tem 1 irmão. remove os dois                   
                    if len(siblings) == 1:
                        # Check if the sum of children's repo_count is below the threshold
                        self.parent.children = []
                        self.parent.is_leaf = True
                        self.id = None  # Reset ID until reset_ids is called
                        self.repo_count - self.parent.repo_count

                        break

                    else:

                        # Find the adjacent sibling with the lowest repo_count
                        adjacent_sibling = None
                        min_count = float('inf')

                        for sibling in siblings:
                            # print(self.is_adjacent(sibling))
                            if self.is_adjacent(sibling) and sibling.repo_count < min_count:
                                adjacent_sibling = sibling
                                min_count = sibling.repo_count

                        # Merge with the adjacent sibling
                        self.x_min = min(self.x_min, adjacent_sibling.x_min)
                        self.x_max = max(self.x_max, adjacent_sibling.x_max)
                        self.y_min = min(self.y_min, adjacent_sibling.y_min)
                        self.y_max = max(self.y_max, adjacent_sibling.y_max)
                        self.repo_count += adjacent_sibling.repo_count
                        mv[adjacent_sibling.id]=self.id
                        
                        if adjacent_sibling.merged: 
                            self.merged = self.merged + adjacent_sibling.merged
                        else:
                            self.merged += 1

                        

                        # Remove merged sibling from parent's children
                        self.parent.children.remove(adjacent_sibling)

                        # count +=1

    def is_adjacent(self, other):
        """
        Check if two nodes are adjacent (sharing an edge but not a diagonal).
        Two nodes are adjacent if they share either the same x-range or y-range
        and their boundaries touch.
        """
        # Check horizontal adjacency (same y-range, x-coordinates are consecutive)
        if self.x_min >= other.x_min and self.x_min <= other.x_max:
            return True
        if self.x_min <= other.x_min and self.x_max >= other.x_max:
            return True

        # Check horizontal adjacency (same y-range, x-coordinates are consecutive)
        if self.y_min >= other.y_min and self.y_min <= other.y_max:
            return True
        if self.y_min <= other.y_min and self.y_max >= other.y_max:
            return True


        return False

    def naive_split(self, tr, mv, k, r, new = False):
        # print("quadtree_id_counter:",QuadTree.id_counter)
        # print("naive_split")
        # print("tr:",tr)
        """Perform naive split based on the threshold `tr` and counts for each cell."""
        if self.is_leaf:
            # print("is_leaf")
            # print("repo_count:",self.repo_count)
            if self.repo_count > tr and r>0:
                # print("repo_count:",self.repo_count)
                # Subdivide the cell if the count exceeds the threshold
                self.subdivide()
                mv[self.id] = [k,k+1,k+2,k+3]
                k = k + 4
                # Redistribute counts to the child nodes
                redistributed_count = self.repo_count // 4
                remain = self.repo_count % 4
                for child in self.children:
                    if remain > 0:
                        child.repo_count = redistributed_count + 1
                        remain -= 1
                    else:
                        child.repo_count = redistributed_count

                # Continue naive_split for the children
                for child in self.children:
                    new = True
                    child.naive_split(tr,mv,k,r-1,new)
            else:
                # print("entrei")
                if new:
                    self.id = QuadTree.id_counter
                    QuadTree.id_counter += 1
                    mv[self.id] = self.id
        else:
            for child in self.children:
                child.naive_split(tr,mv,k,r)

    def set_counts(self, counts):
        """Set repo_count for leaves based on the provided list of counts."""
        leaves = self.get_leaves()
        for leaf, count in zip(leaves, counts):
            leaf.repo_count = count

    def get_leaves(self):
        """Return all leaves in the quad tree."""
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def calculate_repo_counts(self):
        """Calculate the repo_count for all nodes recursively."""
        if self.is_leaf:
            return self.repo_count
        self.repo_count = sum(child.calculate_repo_counts() for child in self.children)
        return self.repo_count

    def display(self, depth=0):
        indent = " " * (depth * 2)
        if self.parent is None:
            parent = "Root"
        else:
            parent = self.parent.global_id
        if self.is_leaf:
            print(f"{indent}Leaf (ID: {self.id},Parent: {parent}, Level: {self.level}, Repo Count: {self.repo_count}): "
                  f"[{self.x_min}, {self.x_max}] x [{self.y_min}, {self.y_max}]")
        else:
            print(f"{indent}Node (Parent: {parent},Level: {self.level}, Repo Count: {self.repo_count}): "
                  f"[{self.x_min}, {self.x_max}] x [{self.y_min}, {self.y_max}]")
            for child in self.children:
                child.display(depth + 1)

    def leaf_ids(self):
        """Return a list of IDs for all leaves in the quad tree."""
        return [leaf.id for leaf in self.get_leaves()]
    
    # go through the tree and update the map_vector
    def update_map_vector(self,mp):
        updated_vector = []
    
        for node_id in mp:
            node = self.find_node_by_id(node_id)  # Lookup node in the quadtree
            if node:
                updated_vector.append(self.resolve_node(node))
            else:
                updated_vector.append(node_id)  # If not found, keep it unchanged
        
        return updated_vector
            
        
    
    def resolve_node(self,node):
        """ Recursively resolves the node to either an ID or a nested list of children """
        if node.is_leaf:
            return node.id  # Return ID if it's a leaf node
        
        # Get child nodes that are not None
        children = [child for child in node.children if child is not None]
        
        if not children:  # If no valid children, return itself
            return node.id
        
        # Recursively resolve child nodes
        return [self.resolve_node(child) for child in children]
    
    def find_node_by_id(self, node_id):
        """ Recursively searches for a node with a given ID """
        if self.id == node_id:
            return self  # Found the node
        
        if self.children:
            for child in self.children:
                if child:
                    found = child.find_node_by_id(node_id)
                    if found:
                        return found  # Return immediately when found
        return None  # Not found
    
    def get_map(self,k):
        # Exemplo de uso
        mapeamento = {}
        for i in range(k):
            mapeamento[i]=i
        return mapeamento

    def modificar_mapeamento(self,mapeamento, id_elemento, operacao, novo_id=None):
        """
        Modifica o mapeamento usando um dicionário.

        Args:
        mapeamento: O dicionário que representa o mapeamento.
        id_elemento: O ID do elemento a ser modificado.
        operacao: A operação a ser realizada ('dividir' ou 'remover').
        novo_id: O novo ID para o qual os ponteiros devem ser ajustados 
                (apenas para a operação 'remover').

        Returns:
        O dicionário modificado.
        """
        if operacao == 'dividir':
            novos_ids = list(range(max(mapeamento.keys(), default=-1) + 1, max(mapeamento.keys(), default=-1) + 5))
            mapeamento[id_elemento] = novos_ids
            for novo_id_elemento in novos_ids:
                mapeamento[novo_id_elemento] = novo_id_elemento

        elif operacao == 'remover':
            del mapeamento[id_elemento]
            for chave, valor in mapeamento.items():
                if isinstance(valor, list):
                    for i in range(len(valor)):
                        if valor[i] == id_elemento:
                            valor[i] = novo_id
                elif valor == id_elemento:
                    mapeamento[chave] = novo_id

        return mapeamento


#Best split first with v2 and after merge, ensuring that we will not have cells with low counts
#  

# if __name__ == "__main__":
    
#     x_min, x_max = 0, 10000
#     y_min, y_max = 0, 10000
#     leaf_size = 2500
#     quad_tree = QuadTree(x_min, x_max, y_min, y_max, leaf_size)

    
#     start_time = time.time()
#     quad_tree.build()
#     quad_tree.reset_ids()


#     print("Initital QUAD-TREE----------------------------------------------------------")
#     quad_tree.display()
#     end_time = time.time()
#     print("t:",end_time-start_time)

#     # Example count list and thresholds
#     counts = [13, 27, 1, 5, 9, 18, 60, 0, 10, 10, 11, 23, 13, 4, 19, 8]
#     threshold_split = 40
#     threshold_merge = 9

#     # # Set initial counts
#     quad_tree.set_counts(counts)
#     quad_tree.calculate_repo_counts()
#     quad_tree.display()

#     map_vector = quad_tree.leaf_ids()

#     print("map_verctor:",map_vector)

#     quad_tree.merge_children_drop_child_v2(threshold_merge)
#     quad_tree.calculate_repo_counts()
#     quad_tree.reset_ids()
#     quad_tree.display()

#     k = len(map_vector)
#     mapeamento = quad_tree.get_map(k)
    
#     quad_tree.naive_split(threshold_split,mapeamento,k,2)
#     quad_tree.calculate_repo_counts()
#     quad_tree.reset_ids()
#     quad_tree.display()


#     # # Perform naive split
#     # quad_tree.naive_split(threshold_split, counts)
#     # quad_tree.calculate_repo_counts()

#     # # Reset IDs after modifications
#     # quad_tree.reset_ids()

#     # # Display tree after split
#     # print("Tree after naive split:")
#     # quad_tree.display()

#     # # Perform merging of nodes
#     # # Example count list and thresholds
#     # print("AFTER SET NEW COUNTS")
#     # counts = [10, 2, 1, 4, 5, 6, 7, 0, 1, 10, 11, 1, 1, 1, 0, 8, 5, 8,0]
#     # quad_tree.set_counts(counts)
#     # quad_tree.calculate_repo_counts()
#     # quad_tree.display()



#     # quad_tree.merge_children(threshold_merge)
#     # quad_tree.calculate_repo_counts()

#     # # Reset IDs after merging
#     # quad_tree.reset_ids()

#     # # Display tree after merging
#     # print("\nTree after merging:")
#     # quad_tree.display()


#     ## TESTE TAM 4 ###########
#     x_min, x_max = 0, 100
#     y_min, y_max = 0, 100
#     leaf_size = 25
#     quad_tree = QuadTree(x_min, x_max, y_min, y_max, leaf_size)

#     start_time = time.time()
#     quad_tree.build()
#     quad_tree.reset_ids_leafs()


#     print("Initital QUAD-TREE----------------------------------------------------------")
#     quad_tree.display()
#     end_time = time.time()
#     print("t:",end_time-start_time)

 
#     print("-----------------------------------------------------------------------------")
#     k = quad_tree.get_cell_count()
#     print("------Leaf_count:",k)

#     mapeamento = quad_tree.get_map(k)
#     print(mapeamento)

#     # ### generate map_vector
#     # map_vector = quad_tree.leaf_ids()
#     # print("------Map_vector:",map_vector)
#     # print("len map vector:",len(map_vector))

#     #Perform merging of nodes
#     #Example count list and thresholds
#     print("AFTER SET NEW COUNTS")
#     counts = [10,120,10,10,10,10,10,10,10,10,10,10,1,1,3,5]
#     quad_tree.set_counts(counts)
#     quad_tree.calculate_repo_counts()
#     quad_tree.display()
    

#     threshold_split = 50
#     threshold_merge = 10

#     # Set initial counts
    
#     # Perform naive split
#     QuadTree.id_counter = k
#     quad_tree.naive_split(threshold_split,mapeamento,k)
#     k = quad_tree.get_cell_count()
#     quad_tree.calculate_repo_counts()
#     quad_tree.display()
#     print("map_vector:",mapeamento)

#     counts = [10,1,1,1,1,10,10,10,10,10,10,10,10,10,10,1,1,3,5]
#     print(len(counts))
#     quad_tree.set_counts(counts)
#     quad_tree.calculate_repo_counts()
#     quad_tree.display()

#     # # Perform naive split
#     # QuadTree.id_counter = k
#     # quad_tree.naive_split(threshold_split,mapeamento,k)
#     # quad_tree.calculate_repo_counts()
#     # quad_tree.display()
#     # print("map_vector:",mapeamento)

#     # # Perform merge:
#     quad_tree.merge_children(threshold_merge,mapeamento)
#     quad_tree.calculate_repo_counts()
#     quad_tree.display()
#     print("map_vector:",mapeamento)

#     # # ### generate map_vector
#     # # # map_vector = quad_tree.leaf_ids()
#     # # # map_vector = quad_tree.update_map_vector(map_vector)
#     # # # Uma vez atualizado. Cada Usuario vai 
   

#     # # # # Reset IDs after modifications
#     # # # quad_tree.reset_ids()
#     # # quad_tree.reset_ids_leafs()

#     # #  ### generate map_vector
#     # # map_vector = quad_tree.leaf_ids()

#     # # #Display tree after split
#     # # print("Tree after naive split:")
#     # # quad_tree.display()
#     # # print("------Map_vector:",map_vector)
    
#     # # quad_tree.merge_children(threshold_merge)
#     # # quad_tree.calculate_repo_counts()

#     # # # # Reset IDs after merging
#     # # # quad_tree.reset_ids()
#     # # quad_tree.reset_ids_leafs()

#     # # map_vector = quad_tree.leaf_ids()

#     # # # Display tree after merging
#     # # print("\nTree after merging:")
#     # # quad_tree.display()
#     # # print("------Map_vector:",map_vector)


#     # # P = (0,100000)
#     # # leaf = quad_tree.find_leaf(*P)

#     # # print("Leaf containing point P:", leaf.id)

