import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Function to add a layer
def add_layer(ax, position, size, text, color='lightblue'):
    """Adds a rectangular layer to the plot."""
    rect = Rectangle(position, size[0], size[1], edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    cx = position[0] + size[0] / 2.0
    cy = position[1] + size[1] / 2.0
    ax.text(cx, cy, text, color='black', weight='bold', fontsize=8, ha='center', va='center')

# Function to add an arrow
def add_arrow(ax, start, end):
    """Adds an arrow to the plot connecting two points."""
    arrow = FancyArrowPatch(start, end, mutation_scale=10, color='black', arrowstyle='-|>')
    ax.add_patch(arrow)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)

# Adding layers
add_layer(ax, (1, 9.5), (8, 2), 'Presentation Layer\n- UI: customtkinter, tkinter\n- Web Interface: Flask\n- Audio: pygame')
add_layer(ax, (1, 6.5), (8, 2.5), 'Application Layer\n- Flask Routes & Business Logic\n- User Authentication\n- Feedback Handling\n- Asynchronous Tasks: threading\n- PDF Handling: fitz\n- Speech Recognition\n- Text-to-Speech: gtts\n- Language Detection\n- AI Processing: openai')
add_layer(ax, (1, 3.5), (8, 2), 'Data Layer\n- Vector Databases: Qdrant\n- Data Processing: pandas\n- File Monitoring: watchdog\n- Caching: functools.lru_cache\n- Logging: logging')
add_layer(ax, (1, 0.5), (8, 2), 'Integration Layer\n- Environment Variables: os\n- Configuration Management: config\n- External API Interactions: requests, openai\n- Embedding & AI Services: openai\n- Database Interactions: QdrantClient')

# Adding arrows for interaction
add_arrow(ax, (5, 9.5), (5, 7.75))  # Presentation to Application
add_arrow(ax, (5, 6.5), (5, 4.75))  # Application to Data
add_arrow(ax, (5, 3.5), (5, 2.25))  # Data to Integration

# Turn off the axes
plt.axis('off')

# Show the plot
plt.show()
