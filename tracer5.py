import pandas as pd
import networkx as nx
from collections import defaultdict, deque
from typing import List, Dict, Tuple
import numpy as np
import random
import base64
from io import BytesIO

# Bokeh imports
from bokeh.plotting import figure, show, save, output_file
from bokeh.models import (HoverTool, MultiLine, Circle, Rect, ColumnDataSource,
                         ColorBar, LinearColorMapper, BasicTicker, ColorBar,
                         Range1d, Label, LabelSet, GraphRenderer, StaticLayoutProvider,
                         EdgesAndLinkedNodes, NodesAndLinkedEdges, TapTool, BoxSelectTool)
from bokeh.palettes import Spectral8, Category20, RdYlBu, Viridis
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.widgets import Div

# RDKit imports for molecular structure visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
    print("RDKit可用 - 将显示分子结构")
except ImportError:
    RDKIT_AVAILABLE = False
    print("警告: RDKit未安装 - 无法显示分子结构")

def smiles_to_base64_image(smiles: str, size=(150, 150)) -> str:
    """
    将SMILES字符串转换为base64编码的PNG图片
    
    Parameters:
    -----------
    smiles : str
        SMILES字符串
    size : tuple
        图片尺寸 (width, height)
    
    Returns:
    --------
    str
        base64编码的图片字符串，如果失败则返回占位符
    """
    if not RDKIT_AVAILABLE or not smiles or smiles == "N/A":
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    try:
        # 从SMILES创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        # 生成2D坐标
        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)
        
        # 生成图片
        img = Draw.MolToImage(mol, size=size, kekulize=True)
        
        # 转换为base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        print(f"生成分子结构图片失败 (SMILES: {smiles}): {e}")
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

def generate_molecule_images(species_dict: Dict[str, str]) -> Dict[str, str]:
    """
    为所有物种生成分子结构图片
    
    Parameters:
    -----------
    species_dict : Dict[str, str]
        物种ID到SMILES的映射
    
    Returns:
    --------
    Dict[str, str]
        物种ID到base64图片的映射
    """
    print("正在生成分子结构图片...")
    molecule_images = {}
    
    for species_id, smiles in species_dict.items():
        img_base64 = smiles_to_base64_image(smiles, size=(120, 120))  # 小尺寸图片
        molecule_images[species_id] = img_base64
    
    print(f"完成生成 {len(molecule_images)} 个分子结构图片")
    return molecule_images

# 假设 Reaction 类已经从 reaction.py 导入
# from reaction import Reaction, get_all_reactions

# ============================================================================
# 分子结构可视化功能说明:
# 
# 1. 安装RDKit: pip install rdkit
# 2. 功能: 悬停物种节点时显示分子结构图片（120x120像素）
# 3. 数据源: 从SMILES字符串通过RDKit生成PNG图片
# 4. 容错: 如果RDKit未安装或SMILES无效，显示占位符图片
# 5. 性能: 图片在网络构建时预生成，悬停时直接显示
# ============================================================================

def set_random_seeds(seed=42):
    """设置随机种子确保布局一致性"""
    random.seed(seed)
    np.random.seed(seed)

class Reaction:
    """简化的 Reaction 类定义（如果需要独立运行）"""
    def __init__(self, Rx, reaction_S, reaction_formula, reaction_smiles,
                 time_profile, k, k_r):
        self.Rx = Rx
        self.reaction_S = reaction_S
        self.reaction_formula = reaction_formula
        self.reaction_smiles = reaction_smiles
        self.time_profile = time_profile
        self.reactants_formula = [
            i.strip() for i in reaction_formula.split('->')[0].split('+')]
        self.reactants_smiles = [i.strip()
                                 for i in reaction_smiles.split(':')[0].split(',')]
        self.num_reactants = len(self.reactants_formula)
        self.products_formula = [
            i.strip() for i in reaction_formula.split('->')[1].split('+')]
        self.products_smiles = [i.strip()
                                for i in reaction_smiles.split(':')[1].split(',')]
        self.k = k
        self.k_r = k_r
        self.occurences = None
    
    def occur_at(self):
        time_non_zero = self.time_profile[0][self.time_profile[1] != 0]
        occurrences = self.time_profile[1][self.time_profile[1] != 0]
        return np.array([time_non_zero, occurrences])

def get_all_reactions(Path_to_reaction_data: str) -> List[Reaction]:
    """从csv文件中读取反应数据，并创建Reaction实例（跳过速率数据）"""
    reaction_data = pd.read_csv(Path_to_reaction_data, header=None, low_memory=False)
    reactions = []
    time_steps = np.array(reaction_data.iloc[4:, 0], dtype=float)
    time_steps = np.array(time_steps, dtype=int)
    
    # 按列排序处理以确保一致性
    for col in sorted(reaction_data.columns[1:]):
        Rx = reaction_data[col][0]
        reaction_S = reaction_data[col][1]
        reaction_formula = reaction_data[col][2]
        reaction_smiles = reaction_data[col][3]
        num_profile = np.array(reaction_data[col][4:], dtype=float)
        num_profile = np.array(num_profile, dtype=int)
        time_profile = np.array([time_steps, num_profile], dtype=int)
        
        # 跳过速率读取，设置默认值
        k = 0.0
        k_r = 0.0
        
        reaction = Reaction(Rx, reaction_S, reaction_formula,
                            reaction_smiles, time_profile, k, k_r)
        reaction.occurences = reaction.occur_at()
        reactions.append(reaction)
    return reactions

def calculate_directional_layout(graph, highlight_paths=None):
    """
    计算具有方向性的分层布局 - 从左到右清晰呈现路径
    """
    print(f"计算分层方向性布局，节点数: {len(graph.nodes())}")
    
    # 分离物种节点和反应节点
    species_nodes = [n for n in graph.nodes() if graph.nodes[n]['node_type'] == 'species']
    reaction_nodes = [n for n in graph.nodes() if graph.nodes[n]['node_type'] == 'reaction']
    
    # 如果有高亮路径，基于路径计算层级
    if highlight_paths and len(highlight_paths) > 0:
        pos = calculate_path_based_layout(graph, highlight_paths, species_nodes, reaction_nodes)
    else:
        # 备用方案：基于网络拓扑计算层级
        pos = calculate_topology_based_layout(graph, species_nodes, reaction_nodes)
    
    # 在保持方向性的基础上，适度分散相近节点
    pos = spread_nodes_in_layers(pos, min_distance=0.3)
    
    return pos

def calculate_path_based_layout(graph, highlight_paths, species_nodes, reaction_nodes):
    """基于高亮路径计算分层布局"""
    print("基于路径计算分层布局")
    
    # 收集所有路径中的节点和它们的位置
    path_positions = {}
    max_path_length = 0
    
    for path in highlight_paths:
        for i, node in enumerate(path):
            if node not in path_positions:
                path_positions[node] = []
            path_positions[node].append(i)
        max_path_length = max(max_path_length, len(path))
    
    pos = {}
    
    # 为路径中的节点分配位置
    for node in graph.nodes():
        if node in path_positions:
            # 使用平均位置作为x坐标（层级）
            avg_position = sum(path_positions[node]) / len(path_positions[node])
            x = avg_position / max(max_path_length - 1, 1)  # 归一化到[0,1]
        else:
            # 非路径节点放在右侧
            x = 1.2
        
        # 根据节点类型设置初始y坐标
        if graph.nodes[node]['node_type'] == 'species':
            y = 0.5 + np.random.normal(0, 0.2)  # 物种节点在中心线附近
        else:
            y = 0.3 + np.random.normal(0, 0.15)  # 反应节点稍微偏下
        
        pos[node] = (x, y)
    
    return pos

def calculate_topology_based_layout(graph, species_nodes, reaction_nodes):
    """基于网络拓扑计算分层布局"""
    print("基于拓扑结构计算分层布局")
    
    # 找到源节点（入度为0的物种节点）
    source_candidates = [n for n in species_nodes if graph.in_degree(n) == 0]
    if not source_candidates:
        source_candidates = species_nodes[:1]  # 备用：选择第一个物种节点
    
    # 计算每个节点到源节点的最短距离
    distances = {}
    for source in source_candidates:
        try:
            dist = nx.single_source_shortest_path_length(graph, source)
            for node, d in dist.items():
                if node not in distances or d < distances[node]:
                    distances[node] = d
        except:
            continue
    
    # 为没有距离的节点分配默认距离
    max_dist = max(distances.values()) if distances else 0
    for node in graph.nodes():
        if node not in distances:
            distances[node] = max_dist + 1
    
    # 根据距离分配x坐标（层级）
    max_distance = max(distances.values())
    pos = {}
    
    for node in graph.nodes():
        # x坐标基于距离（从左到右）
        x = distances[node] / max(max_distance, 1) if max_distance > 0 else 0
        
        # y坐标基于节点类型和随机偏移
        if graph.nodes[node]['node_type'] == 'species':
            y = 0.6 + np.random.normal(0, 0.2)  # 物种节点
        else:
            y = 0.4 + np.random.normal(0, 0.15)  # 反应节点
        
        pos[node] = (x, y)
    
    return pos

def spread_nodes_in_layers(pos, min_distance=0.3):
    """在保持x坐标（层级）的基础上，分散y坐标避免重叠"""
    print(f"在分层内部分散节点，最小距离: {min_distance}")
    
    # 按x坐标分组
    layers = {}
    for node, (x, y) in pos.items():
        x_rounded = round(x, 2)  # 允许小的x坐标差异
        if x_rounded not in layers:
            layers[x_rounded] = []
        layers[x_rounded].append((node, x, y))
    
    # 在每一层内部调整y坐标
    new_pos = {}
    for layer_x, nodes_in_layer in layers.items():
        if len(nodes_in_layer) <= 1:
            # 单个节点，直接使用原坐标
            for node, x, y in nodes_in_layer:
                new_pos[node] = (x, y)
        else:
            # 多个节点，需要分散
            nodes_in_layer.sort(key=lambda item: item[2])  # 按y坐标排序
            
            # 计算需要的总高度
            total_height = (len(nodes_in_layer) - 1) * min_distance
            start_y = 0.5 - total_height / 2  # 居中分布
            
            for i, (node, x, y) in enumerate(nodes_in_layer):
                new_y = start_y + i * min_distance
                new_pos[node] = (x, new_y)
    
    return new_pos

class ReactionNetworkBuilder:
    def __init__(self, spec_file: str, reac_file: str):
        self.species_dict = self._read_species_data(spec_file)
        self.reactions = get_all_reactions(reac_file)  # 不再需要rate_file
        self.network = nx.DiGraph()
        
        # 生成分子结构图片
        self.molecule_images = generate_molecule_images(self.species_dict) if RDKIT_AVAILABLE else {}

    def _read_species_data(self, spec_file: str) -> Dict[str, str]:
        """读取物种数据"""
        try:
            df = pd.read_csv(spec_file, header=None, nrows=2)
            species_dict = {}
            # 按列排序处理以确保一致性
            for col in sorted(df.columns[1:]):
                sx = df[col][0]
                smiles = df[col][1]
                if isinstance(sx, str) and isinstance(smiles, str):
                    species_dict[sx] = smiles
            print(f"读取到 {len(species_dict)} 个物种")
            return species_dict
        except Exception as e:
            print(f"读取物种文件错误: {e}")
            return {}

    def build_bipartite_network(self) -> nx.DiGraph:
        """使用 Reaction 对象构建双向图网络"""
        G = nx.DiGraph()
        
        # 按排序顺序添加物种节点
        for sid, smiles in sorted(self.species_dict.items()):
            G.add_node(sid, node_type="species", species_id=sid, smiles=smiles)

        # 使用 Reaction 对象添加反应节点和边
        valid_reactions = 0
        # 对反应按ID排序处理
        sorted_reactions = sorted(self.reactions, key=lambda r: r.Rx)
        
        for reaction in sorted_reactions:
            rid = reaction.Rx  # 直接使用 R1, R2 等
            
            # 从 Reaction 对象获取反应物和产物
            # 使用 reaction_S 字段（如 "S1+S2->S3"），这更符合网络节点命名
            try:
                reactant_species = []
                product_species = []
                
                # 从 reaction_S 解析物种ID（而不是化学式）
                lhs, rhs = reaction.reaction_S.split('->')
                reactant_species = sorted([s.strip() for s in lhs.split('+')])
                product_species = sorted([s.strip() for s in rhs.split('+')])
                
            except Exception as e:
                print(f"解析反应 {reaction.Rx} 出错: {e}")
                continue
                
            # 检查所有物种是否存在于species_dict中
            all_species_exist = True
            for species in reactant_species + product_species:
                if species not in self.species_dict:
                    all_species_exist = False
                    break
            
            if not all_species_exist:
                continue

            # 添加反应节点，使用 Reaction 对象的所有属性
            node_attrs = {
                'node_type': "reaction",
                'reaction_id': reaction.Rx,
                'reaction_S': reaction.reaction_S,
                'formula': reaction.reaction_formula,
                'smiles': reaction.reaction_smiles,
                'reactants': reactant_species,  # 物种ID列表
                'products': product_species,    # 物种ID列表
                'reactants_formula': reaction.reactants_formula,  # 化学式列表
                'products_formula': reaction.products_formula,    # 化学式列表
                'reactants_smiles': reaction.reactants_smiles,    # SMILES列表
                'products_smiles': reaction.products_smiles,      # SMILES列表
                'reaction_type': f"{len(set(reactant_species))}→{len(set(product_species))}",
                'occurences': reaction.occurences
            }
            
            G.add_node(rid, **node_attrs)

            # 添加边：反应物 -> 反应
            for r in reactant_species:
                if r in G:
                    G.add_edge(r, rid, edge_type="reactant", species_id=r)
            
            # 添加边：反应 -> 产物
            for p in product_species:
                if p in G:
                    G.add_edge(rid, p, edge_type="product", species_id=p)
            
            valid_reactions += 1

        print(f"成功构建网络: {len([n for n in G.nodes() if G.nodes[n]['node_type']=='species'])} 物种, {valid_reactions} 反应")
        self.network = G
        return G

    def find_pathways_backward(self, target: str, source: str, max_paths=10, max_length=20) -> List[List[str]]:
        """改进的反向路径搜索，正确处理多反应物反应"""
        target_node = target  # 直接使用 S1, S2 等
        source_node = source  # 单个源物种
        
        if target_node not in self.network:
            print(f"目标物种 {target} 不在图中")
            return []

        # 检查源物种是否在图中
        if source_node not in self.network:
            print(f"警告：源物种 {source} 不在图中")
            return []

        visited_states = set()
        queue = deque()
        # 初始状态：(当前节点, 路径, 需要的物种集合)
        queue.append((target_node, [target_node], frozenset([target])))
        valid_paths = []

        while queue and len(valid_paths) < max_paths:
            node, path, needed_species = queue.popleft()
            
            # 修改后的终止条件：如果源物种在需要的物种集合中
            if source in needed_species:
                # 找到了一条路径
                # 对于单反应物情况：needed_species == {source}
                # 对于多反应物情况：source in needed_species，其他物种需要额外提供
                complete_path = [source] + list(reversed(path))
                valid_paths.append(complete_path)
                continue

            # 避免重复状态 - 对needed_species排序确保一致性
            state_key = (node, tuple(sorted(needed_species)))
            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            # 控制路径长度
            if len(path) // 2 > max_length:
                continue

            # 如果当前是物种节点，寻找能产生它的反应
            if self.network.nodes[node]['node_type'] == 'species':
                for reaction_node in sorted(self.network.predecessors(node)):
                    if self.network.nodes[reaction_node]['node_type'] != 'reaction':
                        continue
                    
                    # 获取该反应的所有反应物
                    reactants = self.network.nodes[reaction_node]['reactants']
                    # 更新需要的物种集合（移除当前物种，添加反应物）
                    current_species = self.network.nodes[node]['species_id']
                    new_needed = (needed_species - {current_species}) | set(reactants)
                    new_path = path + [reaction_node]
                    queue.append((reaction_node, new_path, frozenset(new_needed)))

            # 如果当前是反应节点，需要所有反应物都可用
            elif self.network.nodes[node]['node_type'] == 'reaction':
                reactants = self.network.nodes[node]['reactants']
                unique_reactants = sorted(list(set(reactants)))  # 去重并排序
                
                # 对于多反应物反应，我们需要一次性添加所有反应物到需求集合
                # 然后为第一个反应物创建路径分支，其他反应物作为额外需求
                if unique_reactants:
                    # 选择字典序第一个反应物作为路径延续点
                    first_reactant = unique_reactants[0]
                    first_reactant_node = first_reactant  # 直接使用物种名
                    
                    if first_reactant_node in self.network:
                        # 所有反应物都成为新的需求
                        all_reactants_set = set(unique_reactants)
                        # 从当前需求中移除已经通过此反应产生的物种
                        current_products = self.network.nodes[node]['products']
                        new_needed = (needed_species - set(current_products)) | all_reactants_set
                        
                        new_path = path + [first_reactant_node]
                        queue.append((first_reactant_node, new_path, frozenset(new_needed)))

        valid_paths.sort(key=lambda p: len(p))
        return valid_paths

    def extract_subgraph_from_paths(self, paths: List[List[str]]) -> nx.DiGraph:
        """从路径中提取子图，包含所有相关的节点和边"""
        subgraph_nodes = set()
        
        # 收集所有路径中的节点
        for path in paths:
            for node in path:
                subgraph_nodes.add(node)
                
                # 如果是反应节点，添加所有反应物和产物
                if node in self.network and self.network.nodes[node]['node_type'] == 'reaction':
                    reactants = self.network.nodes[node]['reactants']
                    products = self.network.nodes[node]['products']
                    subgraph_nodes.update(reactants)
                    subgraph_nodes.update(products)
        
        # 创建子图
        subgraph = self.network.subgraph(subgraph_nodes).copy()
        return subgraph

    def visualize_network(self, subgraph: nx.DiGraph = None, 
                         highlight_paths: List[List[str]] = None,
                         output_filename: str = "network_visualization.html",
                         width: int = 1920, height: int = 1080):
        """
        使用Bokeh可视化反应网络 - 分层方向性布局
        
        Parameters:
        -----------
        subgraph : nx.DiGraph, optional
            要可视化的子图，如果为None则可视化整个网络
        highlight_paths : List[List[str]], optional
            要高亮显示的路径列表
        output_filename : str
            输出HTML文件名
        width, height : int
            图表尺寸
        """
        
        # 设置随机种子确保一致性
        set_random_seeds(42)
        
        # 选择要可视化的图
        graph_to_viz = subgraph if subgraph is not None else self.network
        
        if len(graph_to_viz.nodes()) == 0:
            print("图中没有节点，无法可视化")
            return None
            
        node_count = len(graph_to_viz.nodes())
        print(f"可视化网络: {node_count} 节点, {len(graph_to_viz.edges())} 边")
        
        # 使用分层方向性布局
        pos = calculate_directional_layout(graph_to_viz, highlight_paths)
        
        # 计算坐标范围 - 优化为方向性显示
        x_coords = [pos[node][0] for node in pos]
        y_coords = [pos[node][1] for node in pos]
        
        # x方向：适度扩展以显示层级（从左到右）
        x_min, x_max = min(x_coords), max(x_coords)
        x_range = x_max - x_min
        if x_range < 1.0:  # 确保最小宽度
            x_center = (x_min + x_max) / 2
            x_min, x_max = x_center - 0.5, x_center + 0.5
            x_range = 1.0
        
        # 添加20%的边距
        x_margin = x_range * 0.2
        x_min -= x_margin
        x_max += x_margin
        
        # y方向：适度压缩，保持紧凑
        y_min, y_max = min(y_coords), max(y_coords)
        y_range = y_max - y_min
        if y_range < 0.8:  # 确保最小高度
            y_center = (y_min + y_max) / 2
            y_min, y_max = y_center - 0.4, y_center + 0.4
            y_range = 0.8
        
        # y方向添加较小边距
        y_margin = y_range * 0.15
        y_min -= y_margin
        y_max += y_margin
        
        print(f"方向性布局范围: X[{x_min:.2f}, {x_max:.2f}] (宽度{x_max-x_min:.2f}), Y[{y_min:.2f}, {y_max:.2f}] (高度{y_max-y_min:.2f})")
        
        # 分离物种节点和反应节点
        species_nodes = []
        reaction_nodes = []
        
        for node in graph_to_viz.nodes():
            if graph_to_viz.nodes[node]['node_type'] == 'species':
                species_nodes.append(node)
            else:
                reaction_nodes.append(node)
        
        # 创建Bokeh图表 - 优化为宽屏比例显示方向性
        plot = figure(width=width, height=height, 
                     x_range=Range1d(x_min, x_max), y_range=Range1d(y_min, y_max),
                     title="反应网络可视化 (分层方向性布局 - 从左到右)", 
                     toolbar_location="above",
                     tools="pan,wheel_zoom,box_zoom,reset,save,box_select,tap")
        
        # 去除网格和坐标轴
        plot.xgrid.visible = False
        plot.ygrid.visible = False
        plot.xaxis.visible = False
        plot.yaxis.visible = False
        
        # 添加方向指示箭头
        from bokeh.models import Arrow, VeeHead
        arrow = Arrow(end=VeeHead(size=20), x_start=x_min + 0.05, y_start=y_max - 0.05, 
                     x_end=x_min + 0.2, y_end=y_max - 0.05, line_color="blue", line_width=3)
        plot.add_layout(arrow)
        
        # 添加方向标签
        direction_label = Label(x=x_min + 0.25, y=y_max - 0.08, text="反应方向", 
                               text_color="blue", text_font_size="12pt", text_font_style="bold")
        plot.add_layout(direction_label)
        
        # 准备边的数据
        edge_start = []
        edge_end = []
        edge_colors = []
        edge_alphas = []
        edge_widths = []
        
        # 如果有高亮路径，创建路径边集合
        highlight_edges = set()
        if highlight_paths:
            for path in highlight_paths:
                for i in range(len(path) - 1):
                    highlight_edges.add((path[i], path[i + 1]))
        
        for edge in sorted(graph_to_viz.edges()):  # 排序确保一致性
            start_pos = pos[edge[0]]
            end_pos = pos[edge[1]]
            edge_start.append([start_pos[0], end_pos[0]])
            edge_end.append([start_pos[1], end_pos[1]])
            
            # 边的颜色、透明度和宽度
            if highlight_paths and edge in highlight_edges:
                edge_colors.append("red")
                edge_alphas.append(0.9)
                edge_widths.append(4)  # 高亮路径更粗
            else:
                edge_colors.append("gray")
                edge_alphas.append(0.4)
                edge_widths.append(2)  # 普通边较细
        
        # 添加边
        if edge_start:  # 确保有边存在
            edge_source = ColumnDataSource(data=dict(
                xs=edge_start,
                ys=edge_end,
                colors=edge_colors,
                alphas=edge_alphas,
                widths=edge_widths
            ))
            
            multi_line = MultiLine(xs="xs", ys="ys", 
                                  line_color="colors", 
                                  line_alpha="alphas",
                                  line_width="widths")
            plot.add_glyph(edge_source, multi_line)
        
        # 创建高亮节点集合
        highlight_nodes = set()
        if highlight_paths:
            for path in highlight_paths:
                highlight_nodes.update(path)
        
        # 根据布局范围调整节点大小 - 方向性布局需要适中大小
        base_species_size = max(15, min(25, (x_max - x_min) * 8))
        base_reaction_size = max(10, min(15, (x_max - x_min) * 5))
        
        # 添加物种节点
        if species_nodes:
            species_x = [pos[node][0] for node in species_nodes]
            species_y = [pos[node][1] for node in species_nodes]
            species_colors = []
            species_sizes = []
            species_alphas = []
            species_labels = []
            species_smiles = []
            species_images = []  # 添加分子结构图片
            
            for node in species_nodes:
                # 节点颜色和大小
                if node in highlight_nodes:
                    species_colors.append("red")
                    species_sizes.append(base_species_size + 8)
                    species_alphas.append(0.9)
                else:
                    species_colors.append("lightblue")
                    species_sizes.append(base_species_size)
                    species_alphas.append(0.8)
                
                species_labels.append(node)
                species_smiles.append(self.species_dict.get(node, "N/A"))
                
                # 添加分子结构图片
                if RDKIT_AVAILABLE and node in self.molecule_images:
                    species_images.append(self.molecule_images[node])
                else:
                    species_images.append("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
            
            species_source = ColumnDataSource(data=dict(
                x=species_x,
                y=species_y,
                colors=species_colors,
                sizes=species_sizes,
                alphas=species_alphas,
                labels=species_labels,
                smiles=species_smiles,
                images=species_images,  # 添加图片数据
                node_type=["物种"] * len(species_nodes)
            ))
            
            species_circle = Circle(x="x", y="y", size="sizes", 
                                   fill_color="colors", fill_alpha="alphas",
                                   line_color="darkblue", line_width=2)
            species_renderer = plot.add_glyph(species_source, species_circle)
        
        # 添加反应节点
        if reaction_nodes:
            reaction_x = [pos[node][0] for node in reaction_nodes]
            reaction_y = [pos[node][1] for node in reaction_nodes]
            reaction_colors = []
            reaction_sizes = []
            reaction_alphas = []
            reaction_labels = []
            reaction_formulas = []
            reaction_types = []
            reaction_reactants_info = []  # 反应物信息
            reaction_products_info = []   # 产物信息
            
            for node in reaction_nodes:
                node_data = graph_to_viz.nodes[node]
                
                # 节点颜色和大小
                if node in highlight_nodes:
                    reaction_colors.append("orange")
                    reaction_sizes.append(base_reaction_size + 5)
                    reaction_alphas.append(0.9)
                else:
                    reaction_colors.append("lightgreen")
                    reaction_sizes.append(base_reaction_size)
                    reaction_alphas.append(0.8)
                
                reaction_labels.append(node)
                reaction_formulas.append(node_data.get('formula', 'N/A'))
                reaction_types.append(node_data.get('reaction_type', 'N/A'))
                
                # 构建反应物和产物信息
                reactants = node_data.get('reactants', [])
                products = node_data.get('products', [])
                
                reactants_info = " + ".join(reactants) if reactants else "N/A"
                products_info = " + ".join(products) if products else "N/A"
                
                reaction_reactants_info.append(reactants_info)
                reaction_products_info.append(products_info)
            
            reaction_source = ColumnDataSource(data=dict(
                x=reaction_x,
                y=reaction_y,
                sizes=reaction_sizes,
                colors=reaction_colors,
                alphas=reaction_alphas,
                labels=reaction_labels,
                formulas=reaction_formulas,
                types=reaction_types,
                reactants_info=reaction_reactants_info,
                products_info=reaction_products_info,
                node_type=["反应"] * len(reaction_nodes)
            ))
            
            reaction_circle = Circle(x="x", y="y", size="sizes",
                                    fill_color="colors", fill_alpha="alphas",
                                    line_color="darkgreen", line_width=2)
            reaction_renderer = plot.add_glyph(reaction_source, reaction_circle)
        
        # 动态调整标签字体大小
        layout_width = x_max - x_min
        base_font_size = max(9, min(14, layout_width * 8))
        label_font_size = f"{int(base_font_size)}pt"
        
        # 添加标签（优化偏移量）
        if species_nodes:
            label_offset_y = (y_max - y_min) * 0.03
            
            species_label_source = ColumnDataSource(data=dict(
                x=species_x,
                y=[y + label_offset_y for y in species_y],
                labels=species_labels
            ))
            
            species_labels_glyph = LabelSet(x="x", y="y", text="labels",
                                          text_font_size=label_font_size, 
                                          text_align="center",
                                          text_color="darkblue",
                                          text_font_style="bold",
                                          source=species_label_source)
            plot.add_layout(species_labels_glyph)
        
        if reaction_nodes:
            reaction_font_size = f"{max(7, int(base_font_size) - 2)}pt"
            label_offset_y = (y_max - y_min) * 0.025
            
            reaction_label_source = ColumnDataSource(data=dict(
                x=reaction_x,
                y=[y + label_offset_y for y in reaction_y],
                labels=reaction_labels
            ))
            
            reaction_labels_glyph = LabelSet(x="x", y="y", text="labels",
                                           text_font_size=reaction_font_size, 
                                           text_align="center",
                                           text_color="darkgreen",
                                           text_font_style="bold",
                                           source=reaction_label_source)
            plot.add_layout(reaction_labels_glyph)
        
        # 添加hover工具 - 包含分子结构图片，字体加大
        if species_nodes:
            if RDKIT_AVAILABLE:
                # 使用HTML格式显示分子结构图片，字体加大
                species_hover = HoverTool(
                    tooltips="""
                    <div style="width:220px; padding:12px; background-color:white; border:2px solid #333; border-radius:8px;">
                        <div style="font-weight:bold; font-size:18px; color:#333; margin-bottom:10px;">
                            物种: @labels
                        </div>
                        <div style="margin-bottom:10px;">
                            <img src="@images" style="width:120px; height:120px; border:1px solid #ccc; border-radius:4px;">
                        </div>
                        <div style="font-size:14px; color:#666;">
                            <strong>SMILES:</strong><br>
                            <span style="font-family:monospace; word-break:break-all;">@smiles</span>
                        </div>
                    </div>
                    """,
                    renderers=[species_renderer]
                )
            else:
                # 备用方案：不显示图片，字体加大
                species_hover = HoverTool(
                    tooltips=[
                        ("节点ID", "@labels"),
                        ("类型", "@node_type"),
                        ("SMILES", "@smiles")
                    ],
                    renderers=[species_renderer]
                )
            plot.add_tools(species_hover)
        
        if reaction_nodes:
            # 反应节点的悬停工具 - 显示反应详情，字体加大
            reaction_hover = HoverTool(
                tooltips="""
                <div style="width:280px; padding:12px; background-color:#f0f8f0; border:2px solid #2d5a2d; border-radius:8px;">
                    <div style="font-weight:bold; font-size:18px; color:#2d5a2d; margin-bottom:10px;">
                        反应: @labels
                    </div>
                    <div style="font-size:15px; margin-bottom:8px;">
                        <strong>类型:</strong> @types
                    </div>
                    <div style="font-size:14px; margin-bottom:8px;">
                        <strong>反应式:</strong><br>
                        <span style="font-family:monospace;">@formulas</span>
                    </div>
                    <div style="font-size:14px; margin-bottom:6px;">
                        <strong>反应物:</strong> <span style="color:#0066cc;">@reactants_info</span>
                    </div>
                    <div style="font-size:14px;">
                        <strong>产物:</strong> <span style="color:#cc6600;">@products_info</span>
                    </div>
                </div>
                """,
                renderers=[reaction_renderer]
            )
            plot.add_tools(reaction_hover)
        
        # 添加图例说明（更新版本）
        rdkit_status = "可用" if RDKIT_AVAILABLE else "未安装"
        legend_div = Div(text=f"""
        <h3>图例说明 (分层方向性布局)</h3>
        <p><span style="color: lightblue; font-size: 18px;">●</span> 物种节点 - <strong>悬停显示分子结构</strong></p>
        <p><span style="color: lightgreen; font-size: 12px;">●</span> 反应节点 - <strong>悬停显示反应详情</strong></p>
        <p><span style="color: red; font-size: 18px;">●</span> 高亮路径节点</p>
        <p><span style="color: red; font-weight: bold;">━━━</span> 高亮路径边</p>
        <p><span style="color: blue;">➤</span> <b>反应方向: 从左到右</b></p>
        <p><b>节点数:</b> {node_count}</p>
        <p><b>RDKit状态:</b> {rdkit_status}</p>
        <p><b>操作:</b> 悬停查看结构 | 滚轮缩放 | 拖拽平移</p>
        """, width=320, height=220)
        
        # 创建布局
        layout = column(plot, legend_div)
        
        # 设置输出并显示
        output_file(output_filename)
        save(layout)
        print(f"可视化已保存到: {output_filename}")
        
        return plot

    def display_path_details(self, paths: List[List[str]], source: str, target: str):
        """显示路径的详细信息，改进展示方式"""
        if not paths:
            print("未找到有效路径")
            return
            
        print(f"\n找到 {len(paths)} 条从 {source} 到 {target} 的路径:")
        
        for i, path in enumerate(paths):
            print(f"\n=== 路径 {i+1} ===")
            
            # 构建改进的路径序列
            path_parts = []
            j = 0
            while j < len(path):
                if j < len(path) and path[j].startswith('S'):
                    current_species = path[j]
                    
                    # 检查下一个是否是反应节点
                    if j + 1 < len(path) and path[j + 1].startswith('R'):
                        reaction_node = path[j + 1]
                        reaction_data = self.network.nodes[reaction_node]
                        reactants = reaction_data['reactants']
                        
                        # 如果是多反应物反应
                        if len(reactants) > 1:
                            # 找出除了当前物种外的其他反应物
                            other_reactants = [r for r in reactants if r != current_species]
                            if other_reactants:
                                # 构建反应物字符串，额外的反应物用*标记
                                reactant_str = current_species + " + " + " + ".join([f"*{r}" for r in other_reactants])
                                path_parts.append(reactant_str)
                            else:
                                path_parts.append(current_species)
                        else:
                            path_parts.append(current_species)
                        
                        # 添加反应节点
                        path_parts.append(reaction_node)
                        j += 2
                    else:
                        path_parts.append(current_species)
                        j += 1
                else:
                    j += 1
            
            # 添加最后的目标物种（如果不在路径中）
            if len(path_parts) > 0 and not path_parts[-1].startswith('S'):
                path_parts.append(target)
            
            print(f"主路径: {' -> '.join(path_parts)}")
            
            # 收集所有需要的额外反应物
            extra_species = set()
            for node in path:
                if node.startswith('R'):
                    reaction_data = self.network.nodes[node]
                    reactants = reaction_data['reactants']
                    # 找出不在主路径上的反应物
                    for r in reactants:
                        if r != source and r not in path:
                            extra_species.add(r)
            
            if extra_species:
                print(f"\n额外需要的物种: {', '.join(sorted(extra_species))}")

    def analyze_and_visualize_pathways(self, source: str, target: str, 
                                      max_paths: int = 3, max_length: int = 50,
                                      output_filename: str = None):
        """
        综合分析和可视化路径 - 修改为只显示少量路径
        
        Parameters:
        -----------
        source : str
            源物种ID
        target : str  
            目标物种ID
        max_paths : int
            最大路径数量 - 默认减少到3条
        max_length : int
            最大路径长度
        output_filename : str, optional
            输出文件名，如果为None则自动生成
        """
        
        print(f"\n=== 路径分析 ===")
        print(f"从 {source} 到 {target} 的路径搜索")
        
        # 寻找路径
        paths = self.find_pathways_backward(target, source, max_paths=max_paths, max_length=max_length)
        
        if not paths:
            print("未找到有效路径")
            return None, None
        
        # 显示路径详情
        self.display_path_details(paths, source, target)
        
        # 提取子图
        subgraph = self.extract_subgraph_from_paths(paths)
        print(f"\n提取的子图包含: {len(subgraph.nodes())} 节点, {len(subgraph.edges())} 边")
        
        # 生成输出文件名
        if output_filename is None:
            output_filename = f"pathway_{source}_to_{target}.html"
        
        # 可视化 - 使用全屏尺寸
        print(f"\n=== 创建可视化 ===")
        plot = self.visualize_network(
            subgraph=subgraph,
            highlight_paths=paths,
            output_filename=output_filename,
            width=1920,
            height=1080
        )
        
        return paths, subgraph

def main():
    # 设置随机种子确保一致性
    set_random_seeds(42)
    
    # 文件路径
    spec_file = "pyrrole_continue.spec.csv"
    reac_file = "pyrrole_continue.reac.csv"
    
    # 构建网络
    print("=== 构建反应网络 ===")
    builder = ReactionNetworkBuilder(spec_file, reac_file)
    builder.build_bipartite_network()
    
    # 路径搜索和可视化
    source = "S3166"  # 源物种
    target = "S1753"  # 目标物种
    
    # 综合分析和可视化 - 使用分层方向性布局
    paths, subgraph = builder.analyze_and_visualize_pathways(
        source=source, 
        target=target,
        max_paths=3,
        max_length=50,
        output_filename="reaction_network_directional.html"
    )
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()