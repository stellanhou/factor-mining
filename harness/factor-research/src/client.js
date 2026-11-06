// Plain JavaScript client bundle in DSH's public lazy-CJS module format.
window.__ModuleLoader__.load({
  id: 'factor-research-dsh-bundle',
  factory(require) {
    const React = require('react')
    const h = React.createElement
    const inject = ['slots', 'sessions', 'uiWorkspace']
    function apply(ctx) {
      const binding = globalThis.__FACTOR_RESEARCH_UI__
      if (!binding?.workspaceId || !binding.sessionRoles?.coordinator
        || (binding.state !== 'setup' && (!binding.sessionRoles.A || !binding.sessionRoles.B))) {
        throw new Error('factor-research: missing fixed Web Session binding')
      }
      const roles = binding.sessionRoles
      const labels = { coordinator: '研究协调', A: 'A 研究（只读）', B: 'B 验证（只读）' }
      const buttonStyle = { padding: '6px 12px', border: '1px solid var(--dsw-alias-border-l2)',
        borderRadius: 8, background: 'var(--dsw-specific-input-major)', color: 'inherit', cursor: 'pointer' }
      function Navigation({ open, stop, selection, useSessionStatus, useWorkspaces, usePanelInfo }) {
        const statuses = useSessionStatus(value => value)
        const selected = React.useSyncExternalStore(selection.subscribe, selection.getSnapshot)
        const workspace = useWorkspaces(value => value.items.find(item => item.workspaceId === binding.workspaceId))
        const panel = usePanelInfo(value => value.activePanelId)
        const [notice, setNotice] = React.useState('')
        const [stopping, setStopping] = React.useState(false)
        if (panel !== null || !workspace?.sessionIds.includes(selected?.sessionId)
          || !Object.values(roles).includes(selected?.sessionId)) return null
        return h('nav', { 'aria-label': '固定研究会话', style: { display: 'flex', alignItems: 'center',
          flexWrap: 'wrap', gap: 8, padding: '8px 16px', borderTop: '1px solid var(--dsw-alias-border-l2)' } },
        h('span', { title: binding.researchId }, binding.state === 'setup' ? '因子挖掘 · 待绑定研究目标与数据' : '因子研究'),
        ...Object.entries(roles).map(([role, id]) => h('button', { key: role, style: buttonStyle,
          onClick: () => open(id) }, `${labels[role]}${statuses.get(id)?.running ? ' · 运行中' : ''}`)),
        binding.state === 'setup' ? null : h('button', { style: buttonStyle, disabled: stopping, onClick: async () => {
          setStopping(true)
          setNotice('正在停止全部研究会话与计算…')
          try {
            const result = await stop()
            if (!result.ok) throw new Error(`${result.error.code}: ${result.error.message}`)
            // The native cancel RPC acknowledges transport only. The Host keeps
            // stopping until Agent and domain-action quiescence is confirmed.
            setNotice('停止请求已处理；在协调会话核对研究状态后恢复。')
          } catch (error) { setNotice(String(error)) }
          finally { setStopping(false) }
        } }, stopping ? '正在停止…' : '停止研究'),
        h('span', { role: 'status', style: { fontSize: 12 } }, notice))
      }
      ctx.slots.inject('shell.bottom', () => ctx.slots.register({
        name: 'shell.bottom',
        inject: () => ({
          open: id => ctx.uiWorkspace.openSession(id),
          selection: ctx.uiWorkspace.selection,
          stop: () => ctx.sessions.using(roles.coordinator, { source: 'factorResearchStop' },
            reference => reference.binding.session.cancel()),
        }),
      }, Navigation))
      ctx.slots.inject('conversation.composer', () => ctx.slots.register({
        name: 'conversation.composer', priority: -100,
        select: owner => owner.sessionId === roles.A ? 'A'
          : owner.sessionId === roles.B ? 'B' : null,
      }, ({ matched }) => h('p', { style: { textAlign: 'center', padding: 16 }, role: 'note' },
        `${matched} 会话仅供观察。请在“研究协调”中输入或恢复研究。`)))
    }
    return { inject, apply }
  },
})
