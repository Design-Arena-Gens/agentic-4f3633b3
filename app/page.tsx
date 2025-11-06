'use client'

import { useState, useEffect, useRef } from 'react'

interface Neuron {
  id: string
  x: number
  y: number
  active: boolean
  activation: number
  layer: number
}

interface Connection {
  from: string
  to: string
  weight: number
  active: boolean
}

export default function SparseLLM() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [neurons, setNeurons] = useState<Neuron[]>([])
  const [connections, setConnections] = useState<Connection[]>([])
  const [inputText, setInputText] = useState('')
  const [sparsityLevel, setSparsityLevel] = useState(0.85)
  const [isProcessing, setIsProcessing] = useState(false)
  const [stats, setStats] = useState({ active: 0, total: 0, sparsity: 0 })

  // Initialize network architecture
  useEffect(() => {
    const layers = [16, 64, 128, 128, 64, 16] // Encoder-decoder style
    const neuronList: Neuron[] = []
    const connectionList: Connection[] = []

    const width = 1000
    const height = 600
    const layerSpacing = width / (layers.length + 1)

    // Create neurons
    layers.forEach((count, layerIdx) => {
      const verticalSpacing = height / (count + 1)
      for (let i = 0; i < count; i++) {
        neuronList.push({
          id: `L${layerIdx}-N${i}`,
          x: layerSpacing * (layerIdx + 1),
          y: verticalSpacing * (i + 1),
          active: false,
          activation: 0,
          layer: layerIdx
        })
      }
    })

    // Create sparse connections (only connect subset of neurons)
    layers.forEach((count, layerIdx) => {
      if (layerIdx < layers.length - 1) {
        const currentLayer = neuronList.filter(n => n.layer === layerIdx)
        const nextLayer = neuronList.filter(n => n.layer === layerIdx + 1)

        currentLayer.forEach(from => {
          // Each neuron connects to only k random neurons in next layer (sparse connectivity)
          const k = Math.min(8, nextLayer.length)
          const targets = [...nextLayer].sort(() => Math.random() - 0.5).slice(0, k)

          targets.forEach(to => {
            connectionList.push({
              from: from.id,
              to: to.id,
              weight: Math.random() * 2 - 1,
              active: false
            })
          })
        })
      }
    })

    setNeurons(neuronList)
    setConnections(connectionList)
  }, [])

  // Draw network
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw connections
    connections.forEach(conn => {
      const from = neurons.find(n => n.id === conn.from)
      const to = neurons.find(n => n.id === conn.to)
      if (!from || !to) return

      ctx.beginPath()
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(to.x, to.y)

      if (conn.active) {
        ctx.strokeStyle = `rgba(0, 255, 150, ${Math.abs(conn.weight) * 0.6})`
        ctx.lineWidth = 2
      } else {
        ctx.strokeStyle = 'rgba(60, 60, 70, 0.15)'
        ctx.lineWidth = 0.5
      }
      ctx.stroke()
    })

    // Draw neurons
    neurons.forEach(neuron => {
      ctx.beginPath()
      const radius = neuron.active ? 6 : 3
      ctx.arc(neuron.x, neuron.y, radius, 0, Math.PI * 2)

      if (neuron.active) {
        const intensity = neuron.activation
        ctx.fillStyle = `rgba(0, 255, 150, ${intensity})`
        ctx.shadowBlur = 15
        ctx.shadowColor = `rgba(0, 255, 150, ${intensity})`
      } else {
        ctx.fillStyle = 'rgba(80, 80, 90, 0.3)'
        ctx.shadowBlur = 0
      }

      ctx.fill()
      ctx.shadowBlur = 0
    })

    // Draw layer labels
    ctx.fillStyle = '#888'
    ctx.font = '12px monospace'
    const layerNames = ['Input', 'Encode-1', 'Encode-2', 'Bottleneck', 'Decode-1', 'Output']
    const layers = [16, 64, 128, 128, 64, 16]
    layers.forEach((_, idx) => {
      const x = (canvas.width / (layers.length + 1)) * (idx + 1)
      ctx.fillText(layerNames[idx], x - 30, 20)
    })

  }, [neurons, connections])

  // Simulate sparse activation
  const processInput = async (text: string) => {
    if (!text || isProcessing) return

    setIsProcessing(true)

    // Tokenize input (simplified - each char is a feature)
    const features = text.toLowerCase().split('').map(c => c.charCodeAt(0))

    // Reset all neurons
    let updatedNeurons = neurons.map(n => ({
      ...n,
      active: false,
      activation: 0
    }))

    let updatedConnections = connections.map(c => ({
      ...c,
      active: false
    }))

    setNeurons(updatedNeurons)
    setConnections(updatedConnections)

    await new Promise(resolve => setTimeout(resolve, 200))

    // Layer-by-layer forward pass with sparse activation
    for (let layerIdx = 0; layerIdx < 6; layerIdx++) {
      const currentLayerNeurons = updatedNeurons.filter(n => n.layer === layerIdx)

      if (layerIdx === 0) {
        // Input layer - activate based on input features
        currentLayerNeurons.forEach((neuron, idx) => {
          const featureVal = features[idx % features.length] / 128
          neuron.activation = featureVal
          // Sparse: only activate if above threshold
          neuron.active = Math.random() > sparsityLevel
          if (neuron.active) {
            neuron.activation = Math.min(1, featureVal * (1 + Math.random() * 0.5))
          }
        })
      } else {
        // Hidden/Output layers - sparse activation
        const prevLayerNeurons = updatedNeurons.filter(n => n.layer === layerIdx - 1 && n.active)

        currentLayerNeurons.forEach(neuron => {
          let sum = 0
          const incomingConns = updatedConnections.filter(c => c.to === neuron.id)

          incomingConns.forEach(conn => {
            const fromNeuron = updatedNeurons.find(n => n.id === conn.from)
            if (fromNeuron && fromNeuron.active) {
              sum += fromNeuron.activation * conn.weight
              conn.active = true
            }
          })

          // ReLU activation
          const activation = Math.max(0, sum)

          // TOP-K SPARSITY: Only activate neurons with highest activations
          neuron.activation = activation
        })

        // Apply top-k sparsity - only keep top neurons active
        const k = Math.ceil(currentLayerNeurons.length * (1 - sparsityLevel))
        currentLayerNeurons
          .sort((a, b) => b.activation - a.activation)
          .forEach((neuron, idx) => {
            if (idx < k && neuron.activation > 0.1) {
              neuron.active = true
              neuron.activation = Math.min(1, neuron.activation)
            } else {
              neuron.active = false
            }
          })
      }

      setNeurons([...updatedNeurons])
      setConnections([...updatedConnections])
      await new Promise(resolve => setTimeout(resolve, 300))
    }

    // Calculate statistics
    const activeCount = updatedNeurons.filter(n => n.active).length
    const totalCount = updatedNeurons.length
    const actualSparsity = 1 - (activeCount / totalCount)

    setStats({
      active: activeCount,
      total: totalCount,
      sparsity: Math.round(actualSparsity * 100)
    })

    setIsProcessing(false)
  }

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#fff',
      padding: '20px',
      boxSizing: 'border-box'
    }}>
      <div style={{
        maxWidth: '1200px',
        width: '100%'
      }}>
        <h1 style={{
          textAlign: 'center',
          marginBottom: '10px',
          fontSize: '2.5rem',
          background: 'linear-gradient(90deg, #00ff96, #00d4ff)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          fontWeight: 'bold'
        }}>
          Sparse LLM Architecture
        </h1>

        <p style={{
          textAlign: 'center',
          color: '#888',
          marginBottom: '20px',
          fontSize: '0.95rem'
        }}>
          Neurons fire only when needed • Top-K sparse activation • Dynamic routing
        </p>

        <div style={{
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '12px',
          padding: '20px',
          marginBottom: '20px',
          border: '1px solid rgba(255,255,255,0.1)'
        }}>
          <canvas
            ref={canvasRef}
            width={1000}
            height={600}
            style={{
              width: '100%',
              height: 'auto',
              borderRadius: '8px'
            }}
          />
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '20px',
          marginBottom: '20px'
        }}>
          <div style={{
            background: 'rgba(255,255,255,0.05)',
            borderRadius: '12px',
            padding: '20px',
            border: '1px solid rgba(255,255,255,0.1)'
          }}>
            <label style={{ display: 'block', marginBottom: '10px', color: '#aaa' }}>
              Input Text:
            </label>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && processInput(inputText)}
              placeholder="Type something to see sparse activation..."
              disabled={isProcessing}
              style={{
                width: '100%',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid rgba(255,255,255,0.2)',
                background: 'rgba(0,0,0,0.3)',
                color: '#fff',
                fontSize: '1rem',
                boxSizing: 'border-box'
              }}
            />
            <button
              onClick={() => processInput(inputText)}
              disabled={isProcessing || !inputText}
              style={{
                width: '100%',
                marginTop: '10px',
                padding: '12px',
                borderRadius: '8px',
                border: 'none',
                background: isProcessing ? '#555' : 'linear-gradient(90deg, #00ff96, #00d4ff)',
                color: isProcessing ? '#aaa' : '#000',
                fontSize: '1rem',
                fontWeight: 'bold',
                cursor: isProcessing || !inputText ? 'not-allowed' : 'pointer',
                transition: 'all 0.3s'
              }}
            >
              {isProcessing ? 'Processing...' : 'Process Input'}
            </button>
          </div>

          <div style={{
            background: 'rgba(255,255,255,0.05)',
            borderRadius: '12px',
            padding: '20px',
            border: '1px solid rgba(255,255,255,0.1)'
          }}>
            <label style={{ display: 'block', marginBottom: '10px', color: '#aaa' }}>
              Sparsity Level: {Math.round(sparsityLevel * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={sparsityLevel * 100}
              onChange={(e) => setSparsityLevel(parseInt(e.target.value) / 100)}
              disabled={isProcessing}
              style={{
                width: '100%',
                marginBottom: '20px'
              }}
            />

            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr 1fr',
              gap: '10px',
              fontSize: '0.9rem'
            }}>
              <div style={{
                background: 'rgba(0,255,150,0.1)',
                padding: '10px',
                borderRadius: '6px',
                textAlign: 'center'
              }}>
                <div style={{ color: '#00ff96', fontSize: '1.5rem', fontWeight: 'bold' }}>
                  {stats.active}
                </div>
                <div style={{ color: '#888', fontSize: '0.8rem' }}>Active</div>
              </div>
              <div style={{
                background: 'rgba(255,255,255,0.05)',
                padding: '10px',
                borderRadius: '6px',
                textAlign: 'center'
              }}>
                <div style={{ color: '#aaa', fontSize: '1.5rem', fontWeight: 'bold' }}>
                  {stats.total}
                </div>
                <div style={{ color: '#888', fontSize: '0.8rem' }}>Total</div>
              </div>
              <div style={{
                background: 'rgba(0,212,255,0.1)',
                padding: '10px',
                borderRadius: '6px',
                textAlign: 'center'
              }}>
                <div style={{ color: '#00d4ff', fontSize: '1.5rem', fontWeight: 'bold' }}>
                  {stats.sparsity}%
                </div>
                <div style={{ color: '#888', fontSize: '0.8rem' }}>Sparse</div>
              </div>
            </div>
          </div>
        </div>

        <div style={{
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '12px',
          padding: '15px',
          border: '1px solid rgba(255,255,255,0.1)',
          fontSize: '0.85rem',
          color: '#888'
        }}>
          <strong style={{ color: '#00ff96' }}>Architecture Features:</strong>
          <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
            <li>Top-K Sparse Activation: Only neurons with highest activations fire</li>
            <li>Sparse Connectivity: Each neuron connects to limited neurons in next layer</li>
            <li>Dynamic Routing: Activation pathways change based on input</li>
            <li>Energy Efficient: ~{Math.round(sparsityLevel * 100)}% of neurons remain dormant per inference</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
